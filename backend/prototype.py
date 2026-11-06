"""Phase 2 / 2.5: a standalone RAG prototype, run from the terminal.

Usage:
    python prototype.py                      # run the demo
    python prototype.py --reset              # wipe the database first
    python prototype.py --ask "your question" # ask a single question

Real-world webpage text:
    python prototype.py --remember "Python is widely used for data science."
    python prototype.py --remember "text" --source-url "https://example.com"
    python prototype.py --ask "What is Python used for?"
"""

import argparse

from app import config, llm, memory_store

EXAMPLE_MEMORIES = [
    "I am learning Python because I want to get into data science.",
    "My favourite programming language is Python and I use PyCharm.",
    "I prefer dark mode in every code editor and terminal.",
    "I am allergic to peanuts, so I avoid Thai food and satay.",
    "My dog is named Biscuit and he is a golden retriever born in 2021.",
    "I am saving up to travel to Japan in the summer of 2026.",
    "I take my morning coffee with oat milk and no sugar.",
    "My brother Ravi lives in Berlin and works as a data engineer.",
]


def load_examples() -> None:
    print(f"Storing {len(EXAMPLE_MEMORIES)} example memories in ChromaDB...")
    memory_store.add_memories(EXAMPLE_MEMORIES)
    print(f"Done. Database now holds {memory_store.count_memories()} memories.\n")


def show_all() -> None:
    print("All stored memories:")
    for index, memory in enumerate(memory_store.list_memories(), start=1):
        print(f"  {index}. {memory['text']}")
    print()


def remember(text: str, source_url: str, source_title: str) -> None:
    """Store one memory taken from a webpage."""
    memory_id = memory_store.add_memory(
        text=text, source_url=source_url, source_title=source_title, raw_text=text
    )
    print(f"Stored memory {memory_id[:8]}...")
    if source_url:
        print(f"Source: {source_url}\n")
    else:
        print()


def ask(question: str) -> None:
    print(f"Question: {question}")

    matches = memory_store.search_memories(question)

    print("\nRetrieved memories:")
    if not matches:
        print(f"  (none above the relevance threshold of {config.MIN_SCORE})")
    for index, memory in enumerate(matches, start=1):
        print(f"  {index}. [score: {memory['score']}] {memory['text']}")
        if memory["source_url"]:
            print(f"     from: {memory['source_url']}")

    if not matches:
        print("\nAnswer: I don't have a memory about that.\n")
        return

    answer = llm.generate_answer(question, matches)
    print(f"\nAnswer: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryMind RAG prototype")
    parser.add_argument("--reset", action="store_true", help="delete all memories first")
    parser.add_argument("--ask", type=str, default=None, help="ask a single question")
    parser.add_argument(
        "--remember", type=str, default=None, help="store a memory (e.g. text from a webpage)"
    )
    parser.add_argument(
        "--source-url", type=str, default="", help="webpage the memory was copied from"
    )
    parser.add_argument(
        "--source-title", type=str, default="", help="title of that webpage"
    )
    parser.add_argument(
        "--no-examples", action="store_true", help="do not load the built-in demo memories"
    )
    arguments = parser.parse_args()

    print(f"ChromaDB folder: {config.CHROMA_PATH}")
    print(f"LLM provider:    {config.LLM_PROVIDER}")
    print(f"Min score:       {config.MIN_SCORE}\n")

    if arguments.reset:
        removed = memory_store.delete_all_memories()
        print(f"Cleared {removed} memories.\n")

    if arguments.remember:
        remember(arguments.remember, arguments.source_url, arguments.source_title)
        if not arguments.ask:
            return

    if arguments.ask:
        ask(arguments.ask)
        return

    if not arguments.no_examples:
        load_examples()
        show_all()

    # The main demo question from the project brief.
    ask("Why am I learning Python?")
    # A second question to show it retrieves a different memory.
    ask("What is my dog called?")


if __name__ == "__main__":
    main()
