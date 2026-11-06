"""Profile detailed latency stages for Ask workflow."""
from __future__ import annotations

import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import config, llm, memory_store


def profile_ask(question: str):
    print("=" * 60)
    print(f"PROFILING ASK STAGES FOR: '{question}'")
    print("=" * 60)

    t_full_start = time.perf_counter()

    # 1. Query normalization & embedding
    t0 = time.perf_counter()
    normalized_q, expanded = memory_store.expand_query_typos(question)
    typo_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    queries_to_embed = [question]
    if normalized_q != question:
        queries_to_embed.append(normalized_q)
    q_vecs = memory_store.embed_texts(queries_to_embed)
    embed_ms = (time.perf_counter() - t0) * 1000

    # 2. Semantic & Lexical retrieval
    t0 = time.perf_counter()
    coll = memory_store.get_collection()
    count = coll.count()
    sem_res = coll.query(
        query_embeddings=q_vecs,
        n_results=min(config.HYBRID_SEMANTIC_CANDIDATES, count),
        include=["documents", "metadatas", "distances"],
    )
    semantic_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    lex_res = memory_store.search_lexical_fts(question, limit=config.HYBRID_SEMANTIC_CANDIDATES, expanded_terms=expanded)
    lexical_ms = (time.perf_counter() - t0) * 1000

    # 3. Merging and reranking
    t0 = time.perf_counter()
    matches = memory_store.search_memories(question, top_k=config.TOP_K)
    merge_rerank_ms = (time.perf_counter() - t0) * 1000

    # 4. Context construction & token count
    t0 = time.perf_counter()
    user_prompt = llm.build_user_prompt(question, matches)
    prompt_ms = (time.perf_counter() - t0) * 1000
    prompt_chars = len(user_prompt)
    prompt_tokens_est = len(user_prompt.split()) * 1.3  # rough estimate

    print(f"1. Typo Normalization:           {typo_ms:.2f} ms")
    print(f"2. Query Embedding:               {embed_ms:.2f} ms ({len(queries_to_embed)} vectors)")
    print(f"3. Semantic Retrieval (ChromaDB): {semantic_ms:.2f} ms")
    print(f"4. Lexical Retrieval (SQLite FTS):{lexical_ms:.2f} ms")
    print(f"5. End-to-End Search/Rerank:     {merge_rerank_ms:.2f} ms ({len(matches)} matches)")
    print(f"6. Context Construction:         {prompt_ms:.2f} ms ({prompt_chars} chars, ~{int(prompt_tokens_est)} tokens)")

    # 5. LLM Call - Streaming test for TTFT
    print("\nTesting LLM with streaming for TTFT...")
    client = llm._get_grok_client() if config.LLM_PROVIDER == "grok" else None
    
    ttft_ms = 0.0
    llm_total_ms = 0.0
    tokens_generated = 0
    full_output = []

    if client and config.LLM_PROVIDER == "grok":
        t0 = time.perf_counter()
        stream = client.chat.completions.create(
            model=config.GROK_MODEL,
            messages=[
                {"role": "system", "content": llm.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.2,
        )
        first_token = True
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first_token:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    first_token = False
                tokens_generated += 1
                full_output.append(delta)
        llm_total_ms = (time.perf_counter() - t0) * 1000
        raw_answer = "".join(full_output)
    else:
        t0 = time.perf_counter()
        raw_answer, _ = llm.generate_answer_with_citations(question, matches)
        llm_total_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = llm_total_ms

    # 6. Citation sanitization
    t0 = time.perf_counter()
    from app.memory_store import build_source_results
    max_idx = len(build_source_results(matches))
    cleaned_answer, cited_indices = llm.sanitize_citations(raw_answer, max_idx)
    sanitize_ms = (time.perf_counter() - t0) * 1000

    full_request_ms = (time.perf_counter() - t_full_start) * 1000

    print(f"7. Time to First Token (TTFT):   {ttft_ms:.2f} ms")
    print(f"8. Total LLM Generation Time:    {llm_total_ms:.2f} ms (~{tokens_generated} chunks)")
    print(f"9. Citation Sanitization:        {sanitize_ms:.2f} ms (cited: {cited_indices})")
    print(f"10. Full End-to-End Latency:     {full_request_ms:.2f} ms")
    print("\nAnswer Output Preview:")
    print(cleaned_answer[:250] + "...")


if __name__ == "__main__":
    profile_ask("elon must companies")
