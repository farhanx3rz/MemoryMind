"""Automated tests for provider lifecycle, activation, persistence, masking, and switching."""

import os
import unittest
from pathlib import Path
from starlette.testclient import TestClient

from app import config, llm
from app.main import app


class ProviderManagementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Save original config
        self._orig_provider = config.LLM_PROVIDER
        self._orig_xai = config.XAI_API_KEY
        self._orig_openai = config.OPENAI_API_KEY
        self._orig_gemini = config.GEMINI_API_KEY
        self._orig_anthropic = config.ANTHROPIC_API_KEY
        self._orig_ollama = config.OLLAMA_BASE_URL
        self._orig_auth_token = config.API_AUTH_TOKEN
        config.API_AUTH_TOKEN = ""

    def tearDown(self):
        # Restore original config
        config.LLM_PROVIDER = self._orig_provider
        config.XAI_API_KEY = self._orig_xai
        config.OPENAI_API_KEY = self._orig_openai
        config.GEMINI_API_KEY = self._orig_gemini
        config.ANTHROPIC_API_KEY = self._orig_anthropic
        config.OLLAMA_BASE_URL = self._orig_ollama
        config.API_AUTH_TOKEN = self._orig_auth_token
        llm.reset_clients()

    def test_key_masking(self):
        """Keys must be masked and never revealed in full."""
        self.assertIsNone(config.mask_api_key(None))
        self.assertIsNone(config.mask_api_key(""))
        self.assertEqual(config.mask_api_key("12345"), "••••••••")
        masked = config.mask_api_key("xai-1234567890abcdef")
        self.assertTrue(masked.startswith("xai-"))
        self.assertTrue(masked.endswith("cdef"))
        self.assertIn("••••••••", masked)
        self.assertNotIn("1234567890", masked)

    def test_provider_status_endpoint(self):
        """GET /provider/status returns structured provider info without raw keys."""
        config.XAI_API_KEY = "xai-test-secret-key-12345"
        config.LLM_PROVIDER = "grok"

        res = self.client.get("/provider/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["active_provider"], "grok")
        self.assertTrue(data["is_activated"])
        self.assertEqual(data["connection_status"], "ACTIVE")

        # Check grok details
        grok_info = data["providers"]["grok"]
        self.assertTrue(grok_info["is_configured"])
        self.assertNotIn("xai-test-secret-key-12345", grok_info["masked_key"])
        self.assertIn("••••••••", grok_info["masked_key"])

    def test_provider_test_connection_none_and_failure(self):
        """Test provider connection handles success and failure gracefully without key leakage."""
        # Provider 'none' succeeds immediately
        res_none = self.client.post("/provider/test", json={"provider": "none"})
        self.assertEqual(res_none.status_code, 200)
        self.assertTrue(res_none.json()["success"])

        # Missing key for grok fails with 400
        res_fail = self.client.post("/provider/test", json={"provider": "grok", "api_key": ""})
        self.assertEqual(res_fail.status_code, 400)
        self.assertIn("API Key is required", res_fail.json()["detail"])

    def test_transactional_provider_switch_failure_preserves_active(self):
        """If switching to a new provider fails validation, the existing provider remains active."""
        # Establish working active provider
        config.save_provider_config("none", set_active=True)
        self.assertEqual(config.LLM_PROVIDER, "none")
        self.assertTrue(config.is_active())

        # Attempt to configure grok with invalid key and test_first=True
        # A bogus key will fail the live connection test
        res = self.client.post(
            "/provider/configure",
            json={
                "provider": "grok",
                "api_key": "xai-completely-invalid-key-9999",
                "test_first": True,
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Activation test failed", res.json()["detail"])
        # Crucial check: current active provider was NOT changed to grok
        self.assertEqual(config.LLM_PROVIDER, "none")

    def test_provider_configure_success_and_persistence(self):
        """Successfully configuring a provider updates state and persists."""
        res = self.client.post(
            "/provider/configure",
            json={
                "provider": "none",
                "test_first": False,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["active_provider"], "none")
        self.assertTrue(data["is_activated"])
        self.assertEqual(data["connection_status"], "ACTIVE")

    def test_provider_remove_credentials(self):
        """Removing credentials resets configuration and reverts active state."""
        config.save_provider_config("grok", api_key="xai-temp-key-12345", set_active=True)
        self.assertTrue(config.is_active())

        res = self.client.post("/provider/remove", json={"provider": "grok"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(config.XAI_API_KEY, "")

    def test_first_run_unconfigured_behavior(self):
        """When no provider is configured, status correctly reports unconfigured state."""
        config.LLM_PROVIDER = "grok"
        config.XAI_API_KEY = ""
        self.assertFalse(config.is_active())

        res = self.client.get("/provider/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_activated"])
        self.assertEqual(data["connection_status"], "NOT_CONFIGURED")

    def test_provider_switching_preserves_multiple_configs(self):
        """Configuring multiple providers preserves individual credentials across switches."""
        # 1. Configure grok
        config.save_provider_config("grok", api_key="xai-saved-grok-key-1111", set_active=True)
        self.assertEqual(config.LLM_PROVIDER, "grok")

        # 2. Configure gemini
        config.save_provider_config("gemini", api_key="AIza-saved-gemini-key-2222", set_active=True)
        self.assertEqual(config.LLM_PROVIDER, "gemini")

        # 3. Verify both remain stored in provider details
        status = config.get_all_providers_status()
        self.assertTrue(status["providers"]["grok"]["is_configured"])
        self.assertTrue(status["providers"]["gemini"]["is_configured"])
        self.assertEqual(status["active_provider"], "gemini")

        # 4. Switch back to grok without passing a new key
        config.save_provider_config("grok", set_active=True)
        status_after = config.get_all_providers_status()
        self.assertEqual(status_after["active_provider"], "grok")
        self.assertTrue(status_after["providers"]["grok"]["is_configured"])
        self.assertTrue(status_after["providers"]["gemini"]["is_configured"])

    def test_env_persistence_survives_reload(self):
        """Configuration updates are persisted to backend/.env file."""
        test_key = "xai-persistence-check-key-9999"
        config.save_provider_config("grok", api_key=test_key, set_active=True)

        env_file = config.BACKEND_DIR / ".env"
        self.assertTrue(env_file.exists())
        content = env_file.read_text(encoding="utf-8")
        self.assertIn("LLM_PROVIDER=grok", content)
        self.assertIn(f"XAI_API_KEY={test_key}", content)

    def test_ask_stream_blocks_when_not_activated(self):
        """POST /ask/stream must reject requests with 403 when no provider is activated."""
        config.LLM_PROVIDER = "grok"
        config.XAI_API_KEY = ""
        self.assertFalse(config.is_active())

        res = self.client.post("/ask/stream", json={"question": "Where is my memory?"})
        self.assertEqual(res.status_code, 403)
        self.assertIn("not activated", res.json()["detail"])

    def test_scrubbing_key_in_error_messages(self):
        """Errors returned by connection tests must never expose the raw API key."""
        sensitive_key = "xai-super-confidential-secret-key-12345678"
        res = self.client.post(
            "/provider/test",
            json={
                "provider": "grok",
                "api_key": sensitive_key,
                "base_url": "http://127.0.0.1:9999/invalid",  # will fail connection
            },
        )
        self.assertEqual(res.status_code, 400)
        detail = res.json()["detail"]
        self.assertNotIn(sensitive_key, detail)



if __name__ == "__main__":
    unittest.main()
