"""Test synthesis timeout and retry logic."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from app.summarization_jobs import JobManager, JobConfig, SummaryStyle
from app.source_collector import SourceCollection, SourceMetadata, ExtractedSource, SourceType
from app import config


def create_mock_response(content: str) -> ChatCompletion:
    """Create mock OpenAI response."""
    return ChatCompletion(
        id="test",
        object="chat.completion",
        created=1234567890,
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content
                ),
                finish_reason="stop"
            )
        ]
    )


def test_synthesis_timeout_then_success():
    """Test synthesis succeeds after initial timeout."""
    manager = JobManager()
    
    collection = SourceCollection()
    metadata = SourceMetadata(
        source_id="test1",
        source_type=SourceType.WEBPAGE,
        title="Test Article",
        url="https://example.com/test",
        success=True,
    )
    collection.sources.append(ExtractedSource(
        metadata=metadata,
        text="Test content for synthesis"
    ))
    
    system_prompt = "You are a synthesizer."
    user_prompt = "Synthesize this content."
    max_tokens = 500
    
    # Mock client to timeout first, then succeed
    mock_client = Mock()
    call_count = [0]
    
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise APITimeoutError(request=Mock())
        return create_mock_response("Successfully synthesized content")
    
    mock_client.chat.completions.create.side_effect = side_effect
    
    with patch('app.llm._get_grok_synthesis_client', return_value=mock_client):
        result = manager._synthesize_with_grok(system_prompt, user_prompt, max_tokens)
    
    assert result == "Successfully synthesized content"
    assert call_count[0] == 2  # First attempt failed, second succeeded


def test_synthesis_exhausted_retries():
    """Test synthesis fails after exhausting all retries."""
    manager = JobManager()
    
    system_prompt = "You are a synthesizer."
    user_prompt = "Synthesize this content."
    max_tokens = 500
    
    # Mock client to always timeout
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=Mock())
    
    with patch('app.llm._get_grok_synthesis_client', return_value=mock_client):
        with pytest.raises(RuntimeError, match="timed out after .* attempts"):
            manager._synthesize_with_grok(system_prompt, user_prompt, max_tokens)
    
    # Should have tried max_retries times
    assert mock_client.chat.completions.create.call_count == config.SYNTHESIS_MAX_RETRIES


def test_synthesis_permanent_error():
    """Test synthesis fails immediately on permanent error."""
    manager = JobManager()
    
    system_prompt = "You are a synthesizer."
    user_prompt = "Synthesize this content."
    max_tokens = 500
    
    # Mock client to return 400 Bad Request
    mock_response = Mock()
    mock_response.status_code = 400
    mock_error = APIStatusError(
        message="Bad request",
        response=mock_response,
        body={"error": "invalid"}
    )
    
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = mock_error
    
    with patch('app.llm._get_grok_synthesis_client', return_value=mock_client):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            manager._synthesize_with_grok(system_prompt, user_prompt, max_tokens)
    
    # Should only try once for permanent errors
    assert mock_client.chat.completions.create.call_count == 1


def test_synthesis_retryable_status_codes():
    """Test synthesis retries on retryable HTTP status codes."""
    manager = JobManager()
    
    system_prompt = "You are a synthesizer."
    user_prompt = "Synthesize this content."
    max_tokens = 500
    
    for status_code in [429, 500, 502, 503, 504]:
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_error = APIStatusError(
            message=f"HTTP {status_code}",
            response=mock_response,
            body={"error": "temporary"}
        )
        
        mock_client = Mock()
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise mock_error
            return create_mock_response("Success after retry")
        
        mock_client.chat.completions.create.side_effect = side_effect
        
        with patch('app.llm._get_grok_synthesis_client', return_value=mock_client):
            result = manager._synthesize_with_grok(system_prompt, user_prompt, max_tokens)
        
        assert result == "Success after retry"
        assert call_count[0] == 2


def test_synthesis_connection_error_retry():
    """Test synthesis retries on connection errors."""
    manager = JobManager()
    
    system_prompt = "You are a synthesizer."
    user_prompt = "Synthesize this content."
    max_tokens = 500
    
    mock_client = Mock()
    call_count = [0]
    
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise APIConnectionError(request=Mock())
        return create_mock_response("Connected successfully")
    
    mock_client.chat.completions.create.side_effect = side_effect
    
    with patch('app.llm._get_grok_synthesis_client', return_value=mock_client):
        result = manager._synthesize_with_grok(system_prompt, user_prompt, max_tokens)
    
    assert result == "Connected successfully"
    assert call_count[0] == 2


def test_synthesis_rate_limit_retry():
    """Test synthesis retries on rate limit errors."""
    manager = JobManager()
    
    system_prompt = "You are a synthesizer."
    user_prompt = "Synthesize this content."
    max_tokens = 500
    
    mock_client = Mock()
    call_count = [0]
    
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RateLimitError(
                message="Rate limit exceeded",
                response=Mock(status_code=429),
                body={}
            )
        return create_mock_response("Success after rate limit")
    
    mock_client.chat.completions.create.side_effect = side_effect
    
    with patch('app.llm._get_grok_synthesis_client', return_value=mock_client):
        result = manager._synthesize_with_grok(system_prompt, user_prompt, max_tokens)
    
    assert result == "Success after rate limit"
    assert call_count[0] == 2


def test_synthesis_input_size_logging():
    """Test that large input size is logged as warning."""
    manager = JobManager()
    
    # Create large prompt
    system_prompt = "System prompt"
    user_prompt = "x" * (config.SYNTHESIS_MAX_INPUT_CHARS + 1000)
    max_tokens = 500
    
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = create_mock_response("Result")
    
    with patch('app.llm._get_grok_synthesis_client', return_value=mock_client):
        with patch('app.summarization_jobs.logger') as mock_logger:
            result = manager._synthesize_with_grok(system_prompt, user_prompt, max_tokens)
            
            # Check that input size was logged
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("input_chars=" in str(call) for call in info_calls)
            
            # Check success was logged
            assert any("succeeded" in str(call) for call in info_calls)


def test_synthesis_uses_correct_max_tokens():
    """Test that synthesis uses correct max_tokens based on style."""
    manager = JobManager()
    
    collection = SourceCollection()
    metadata = SourceMetadata(
        source_id="test1",
        source_type=SourceType.WEBPAGE,
        title="Test",
        url="https://example.com",
        success=True,
    )
    collection.sources.append(ExtractedSource(metadata=metadata, text="Content"))
    
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = create_mock_response("Result")
    
    # Test QUICK style
    job_config_quick = JobConfig(summary_style=SummaryStyle.QUICK)
    
    with patch('app.llm._get_grok_synthesis_client', return_value=mock_client):
        system_prompt = manager._build_system_prompt(job_config_quick)
        user_prompt = manager._build_user_prompt(collection.get_all_sources(), job_config_quick)
        
        # Should use QUICK max tokens
        manager._synthesize_with_grok(system_prompt, user_prompt, config.SYNTHESIS_MAX_TOKENS_QUICK)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['max_tokens'] == config.SYNTHESIS_MAX_TOKENS_QUICK


def test_openai_synthesis_retry_logic():
    """Test OpenAI synthesis has same retry logic."""
    manager = JobManager()
    
    system_prompt = "You are a synthesizer."
    user_prompt = "Synthesize this content."
    max_tokens = 500
    
    mock_client = Mock()
    call_count = [0]
    
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise APITimeoutError(request=Mock())
        return create_mock_response("OpenAI synthesis success")
    
    mock_client.chat.completions.create.side_effect = side_effect
    
    with patch('app.llm._get_openai_synthesis_client', return_value=mock_client):
        result = manager._synthesize_with_openai(system_prompt, user_prompt, max_tokens)
    
    assert result == "OpenAI synthesis success"
    assert call_count[0] == 2


def test_synthesis_timeout_configuration():
    """Test that synthesis uses configured timeout."""
    from app import llm
    
    # Get synthesis clients
    with patch('app.config.XAI_API_KEY', 'test-key'):
        with patch('app.config.OPENAI_API_KEY', 'test-key'):
            # Reset clients
            llm._grok_synthesis_client = None
            llm._openai_synthesis_client = None
            
            with patch('openai.OpenAI') as mock_openai_class:
                llm._get_grok_synthesis_client()
                
                # Check timeout was set correctly
                call_kwargs = mock_openai_class.call_args[1]
                assert call_kwargs['timeout'] == config.SYNTHESIS_TIMEOUT
                assert call_kwargs['max_retries'] == 0  # We handle retries manually
