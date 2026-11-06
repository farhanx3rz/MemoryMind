import unittest

from app import llm, memory_store


class HybridRetrievalTests(unittest.TestCase):
    def test_keyword_score_rewards_exact_phrase(self):
        score = memory_store._keyword_score(
            "ChromaDB vector database",
            "I use ChromaDB as a vector database for semantic retrieval.",
            "RAG notes",
        )
        self.assertEqual(score, 1.0)

    def test_keyword_score_handles_plural_variation(self):
        score = memory_store._keyword_score(
            "vector databases",
            "ChromaDB is a vector database used for retrieval.",
        )
        self.assertGreaterEqual(score, 0.9)

    def test_tag_score_matches_specific_concept(self):
        tags = [
            {"label": "ChromaDB", "score": 1.0},
            {"label": "Vector Database", "score": 1.0},
        ]
        self.assertEqual(memory_store._tag_score("what did I read about ChromaDB?", tags), 1.0)
        self.assertEqual(memory_store._tag_score("vector databases", tags), 1.0)

    def test_hybrid_score_combines_all_signals(self):
        score = memory_store._hybrid_score(0.8, 0.6, 1.0, 1.0)
        self.assertAlmostEqual(score, 0.81, places=2)


    def test_source_grouping_keeps_multiple_documents_and_best_snippet(self):
        matches = [
            {
                "id": "doc-a:0",
                "document_id": "doc-a",
                "source_title": "Introduction to RAG",
                "source_url": "https://example.com/rag",
                "source_domain": "example.com",
                "visited_at": "2026-10-05T00:14:32+00:00",
                "timestamp": "2026-10-05T00:14:32+00:00",
                "score": 0.94,
                "text": "RAG uses vector databases to retrieve relevant context.",
                "topics": [{"label": "RAG", "score": 1.0}],
                "tags": [{"label": "Vector Database", "score": 1.0}],
                "chunk_index": 0,
            },
            {
                "id": "doc-a:1",
                "document_id": "doc-a",
                "source_title": "Introduction to RAG",
                "source_url": "https://example.com/rag",
                "source_domain": "example.com",
                "visited_at": "2026-10-05T00:14:32+00:00",
                "score": 0.88,
                "text": "Embeddings make semantic retrieval possible.",
                "topics": [{"label": "RAG", "score": 1.0}],
                "tags": [{"label": "Embeddings", "score": 1.0}],
                "chunk_index": 1,
            },
            {
                "id": "doc-b:0",
                "document_id": "doc-b",
                "source_title": "RAG with Vector DBs",
                "source_url": "https://example.org/vector-rag",
                "source_domain": "example.org",
                "visited_at": "2026-10-03T19:42:00+00:00",
                "score": 0.91,
                "text": "A vector database stores embeddings for retrieval.",
                "topics": [{"label": "RAG", "score": 1.0}],
                "tags": [{"label": "Vector Database", "score": 1.0}],
                "chunk_index": 0,
            },
        ]
        sources = memory_store.build_source_results(matches, limit=5)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["document_id"], "doc-a")
        self.assertEqual(sources[0]["chunk_count"], 2)
        self.assertEqual(sources[0]["visited_at"], "2026-10-05T00:14:32+00:00")
        self.assertEqual(sources[0]["snippet"], "RAG uses vector databases to retrieve relevant context.")
        self.assertEqual(
            {tag["label"] for tag in sources[0]["tags"]},
            {"Vector Database", "Embeddings"},
        )


    def test_source_results_keep_same_url_as_separate_document_events(self):
        matches = [
            {
                "id": "doc-a:0",
                "document_id": "doc-a",
                "source_title": "Same Article",
                "source_url": "https://example.com/article",
                "source_domain": "example.com",
                "visited_at": "2026-10-05T00:00:00+00:00",
                "score": 0.95,
                "text": "First capture.",
                "topics": [],
                "tags": [],
                "chunk_index": 0,
            },
            {
                "id": "doc-b:0",
                "document_id": "doc-b",
                "source_title": "Same Article",
                "source_url": "https://example.com/article",
                "source_domain": "example.com",
                "visited_at": "2026-10-05T01:00:00+00:00",
                "score": 0.80,
                "text": "Second capture.",
                "topics": [],
                "tags": [],
                "chunk_index": 0,
            },
        ]
        sources = memory_store.build_source_results(matches, limit=5)
        self.assertEqual(len(sources), 2)
        self.assertNotEqual(sources[0]["source_key"], sources[1]["source_key"])
        self.assertEqual(
            {source["visited_at"] for source in sources},
            {"2026-10-05T00:00:00+00:00", "2026-10-05T01:00:00+00:00"},
        )

    def test_related_concepts_count_distinct_sources(self):
        sources = [
            {
                "document_id": "doc-a",
                "title": "A",
                "url": "https://a.example",
                "score": 0.9,
                "topics": [{"label": "RAG", "score": 1.0}],
                "tags": [{"label": "Vector Database", "score": 1.0}],
            },
            {
                "document_id": "doc-b",
                "title": "B",
                "url": "https://b.example",
                "score": 0.8,
                "topics": [{"label": "RAG", "score": 1.0}],
                "tags": [{"label": "Embeddings", "score": 1.0}],
            },
        ]
        concepts = memory_store.build_related_concepts(sources)
        by_key = {(item["kind"], item["label"]): item for item in concepts}
        self.assertEqual(by_key[("topic", "RAG")]["source_count"], 2)
        self.assertEqual(by_key[("tag", "Vector Database")]["source_count"], 1)

    def test_dynamic_tag_parser_deduplicates_and_bounds_tags(self):
        original_min = llm.config.DYNAMIC_TAG_MIN
        original_max = llm.config.DYNAMIC_TAG_MAX
        try:
            llm.config.DYNAMIC_TAG_MIN = 3
            llm.config.DYNAMIC_TAG_MAX = 5
            tags = llm._parse_dynamic_tags(
                '{"tags":["ChromaDB","Vector Database","ChromaDB","Embeddings","RAG","Too Many"]}'
            )
            self.assertEqual(
                tags,
                ["ChromaDB", "Vector Database", "Embeddings", "RAG", "Too Many"],
            )
        finally:
            llm.config.DYNAMIC_TAG_MIN = original_min
            llm.config.DYNAMIC_TAG_MAX = original_max

    def test_empty_store_returns_empty_list(self):
        from unittest.mock import MagicMock, patch
        mock_coll = MagicMock()
        mock_coll.count.return_value = 0
        with patch.object(memory_store, "get_collection", return_value=mock_coll):
            results = memory_store.search_memories("what did I read about Python?")
            self.assertEqual(results, [])

    def test_search_memories_avoids_full_collection_scan(self):
        from unittest.mock import MagicMock, patch
        mock_coll = MagicMock()
        mock_coll.count.return_value = 500
        mock_coll.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["Python programming language.", "ChromaDB vector store."]],
            "metadatas": [[
                {"source_title": "Python Guide", "document_id": "doc-1", "source_url": "https://python.org"},
                {"source_title": "Chroma Guide", "document_id": "doc-2", "source_url": "https://chroma.org"},
            ]],
            "distances": [[0.1, 0.2]],
        }

        # Track any get() calls that do not provide specific IDs (full collection scan)
        full_collection_scans = []
        def spy_get(*args, **kwargs):
            if not kwargs.get("ids") and (not args or not args[0]):
                full_collection_scans.append(kwargs)
            return {"ids": [], "documents": [], "metadatas": []}

        mock_coll.get.side_effect = spy_get

        with patch.object(memory_store, "get_collection", return_value=mock_coll), \
             patch.object(memory_store, "embed_texts", return_value=[[0.1] * 384]), \
             patch.object(memory_store, "sync_lexical_index", return_value=0), \
             patch.object(memory_store, "search_lexical_fts", return_value={}):
            results = memory_store.search_memories("Python programming", top_k=5)
            self.assertEqual(len(full_collection_scans), 0)
            self.assertGreaterEqual(len(results), 1)

    def test_malformed_metadata_handled_gracefully(self):
        malformed = {
            "topics": "invalid-not-json-or-list",
            "tags": 12345,
        }
        topics = memory_store._parse_topics(malformed)
        tags = memory_store._parse_dynamic_tags(malformed)
        self.assertEqual(topics, [])
        self.assertEqual(tags, [])


if __name__ == "__main__":
    unittest.main()

