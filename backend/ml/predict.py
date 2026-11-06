"""Run the trained classifier from the command line.

Usage:
    python -m ml.predict "A RAG system retrieves documents before generation."
"""
import argparse
import json
from .classifier import TopicClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Article or webpage text")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    classifier = TopicClassifier(threshold=args.threshold) if args.threshold is not None else TopicClassifier()
    print(json.dumps(classifier.predict_tags(args.text), indent=2))


if __name__ == "__main__":
    main()
