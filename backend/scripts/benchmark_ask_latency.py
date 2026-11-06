"""Comprehensive latency benchmark for answer-generation pipeline.

Run from backend/ with: python -c "exec(open('scripts/benchmark_ask_latency.py').read())"
Or: python scripts/benchmark_ask_latency.py
"""
import statistics
import sys
import time

sys.path.insert(0, ".")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import config, llm, memory_store


def time_search(query: str) -> tuple[list, float]:
    """Run search and return (matches, elapsed_ms)."""
    t0 = time.perf_counter()
    matches = memory_store.search_memories(query)
    return matches, (time.perf_counter() - t0) * 1000


def time_llm_stream(question: str, matches: list) -> tuple[float, float, int]:
    """Run LLM streaming and return (ttft_ms, total_ms, token_count)."""
    if not matches:
        return 0.0, 0.0, 0

    prompt = llm.build_user_prompt(question, matches)
    client = llm._get_grok_client()

    t0 = time.perf_counter()
    ttft = None
    tokens = 0

    stream = client.chat.completions.create(
        model=config.GROK_MODEL,
        messages=[
            {"role": "system", "content": llm.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=0.2,
        max_tokens=500,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            tokens += 1

    total = (time.perf_counter() - t0) * 1000
    return ttft or total, total, tokens


def verify_citations(answer: str, matches: list) -> bool:
    """Check if citations in answer reference valid sources."""
    import re

    if not matches:
        return True

    cited = set(re.findall(r"\[(\d+)\]", answer))
    if not cited:
        return False

    max_source = len(matches)
    for c in cited:
        if int(c) > max_source:
            return False
    return True


def run_benchmark(questions: list[str], runs: int = 10):
    """Run comprehensive benchmark."""
    # Pre-warm embedding model
    print("Pre-warming embedding model...")
    _ = memory_store.get_embedding_model()
    print()

    results = {
        "search": [],
        "llm_ttft": [],
        "llm_total": [],
        "total": [],
        "tokens": [],
        "citation_success": [],
    }

    for q in questions:
        print(f"Testing: {q}")
        for i in range(runs):
            matches, search_ms = time_search(q)

            if not matches:
                print(f"  Run {i + 1}: no matches")
                continue

            ttft, llm_ms, tokens = time_llm_stream(q, matches)
            total_ms = search_ms + llm_ms

            # Verify citations
            prompt = llm.build_user_prompt(q, matches)
            client = llm._get_grok_client()
            stream = client.chat.completions.create(
                model=config.GROK_MODEL,
                messages=[
                    {"role": "system", "content": llm.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=500,
            )
            full_answer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_answer += delta

            citation_ok = verify_citations(full_answer, matches)

            results["search"].append(search_ms)
            results["llm_ttft"].append(ttft)
            results["llm_total"].append(llm_ms)
            results["total"].append(total_ms)
            results["tokens"].append(tokens)
            results["citation_success"].append(1 if citation_ok else 0)

            print(
                f"  Run {i + 1}: search={search_ms:.0f}ms, "
                f"TTFT={ttft:.0f}ms, total={total_ms:.0f}ms, "
                f"tokens={tokens}, citations={'OK' if citation_ok else 'FAIL'}"
            )

        print()

    return results


def summarize(name: str, times: list) -> tuple[float, float]:
    """Return (median, p95)."""
    if not times:
        return 0.0, 0.0
    sorted_times = sorted(times)
    median = statistics.median(times)
    p95_idx = max(0, int(len(times) * 0.95) - 1)
    p95 = sorted_times[p95_idx] if sorted_times else 0.0
    return median, p95


def main():
    print("=" * 70)
    print("MemoryMind Answer Pipeline Benchmark")
    print("=" * 70)
    print()

    questions = [
        "elon must companies",
        "what is vector database",
        "python vs javascript",
    ]

    results = run_benchmark(questions, runs=10)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    search_med, search_p95 = summarize("Search", results["search"])
    ttft_med, ttft_p95 = summarize("LLM TTFT", results["llm_ttft"])
    llm_med, llm_p95 = summarize("LLM Total", results["llm_total"])
    total_med, total_p95 = summarize("End-to-End", results["total"])

    citation_rate = (
        sum(results["citation_success"]) / len(results["citation_success"]) * 100
        if results["citation_success"]
        else 0
    )
    avg_tokens = statistics.mean(results["tokens"]) if results["tokens"] else 0

    print(f"{'Metric':<30} {'Median':>12} {'P95':>12}")
    print("-" * 54)
    print(f"{'Search (mem retrieval)':<30} {search_med:>10.1f}ms {search_p95:>10.1f}ms")
    print(f"{'LLM TTFT':<30} {ttft_med:>10.1f}ms {ttft_p95:>10.1f}ms")
    print(f"{'LLM Total Generation':<30} {llm_med:>10.1f}ms {llm_p95:>10.1f}ms")
    print(f"{'End-to-End':<30} {total_med:>10.1f}ms {total_p95:>10.1f}ms")
    print("-" * 54)
    print(f"{'Citation Success Rate':<30} {citation_rate:>11.1f}%")
    print(f"{'Avg Output Tokens':<30} {avg_tokens:>11.1f}")
    print()

    print("BEFORE (original spec):")
    print("  Memory retrieval: 48 ms")
    print("  Full /ask/stream: 13.74 seconds")
    print("  Retrieved chunks: 8")
    print("  Citations: 2")
    print()
    print("AFTER (with optimizations):")
    print(f"  Search median: {search_med:.0f} ms")
    print(f"  LLM TTFT median: {ttft_med:.0f} ms")
    print(f"  End-to-end median: {total_med:.0f} ms")
    print(f"  Citation success: {citation_rate:.0f}%")
    print()

    if total_med < 13740:
        improvement = (13740 - total_med) / 13740 * 100
        print(f"Improvement: {improvement:.1f}% faster than baseline")
    else:
        print("Note: Times include warm model; cold start ~12.7s for first query")


if __name__ == "__main__":
    main()