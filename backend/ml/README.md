# RECALL Topic Classifier

This is the custom ML component of MemoryMind / RECALL. It fine-tunes a pretrained Hugging Face Transformer encoder for **multi-label webpage topic classification**.

It is deliberately separate from the RAG stack:

- **Topic classifier (`distilbert-base-uncased`):** assigns explicit topic/tag probabilities across a 17-label technical taxonomy locally on CPU.
- **Sentence Transformer (`all-MiniLM-L6-v2`):** creates dense 384-dimensional semantic vectors for ChromaDB cosine similarity retrieval.
- **Configured LLM Gateway:** generates the final natural-language answer grounded strictly in retrieved evidence (pre-configured to Grok in `.env.example`, with fallback to Offline / Echo mode when `LLM_PROVIDER` is unconfigured or set to `none`; also supporting Gemini, OpenAI, Anthropic Claude, and Ollama), and can optionally extract dynamic concept tags when enabled.

External LLMs are not used to generate topic taxonomy tags; classification runs 100% locally via the fine-tuned DistilBERT model.

## Model

Default base model: `distilbert-base-uncased`.

The classifier uses a Transformer encoder plus a 17-output classification head with **sigmoid** probabilities and `BCEWithLogitsLoss` through Hugging Face's `multi_label_classification` mode. Softmax is intentionally not used because several topics may be true for one article.

## Labels

The current configurable taxonomy is in `ml/config.py`. It contains 17 topics:

Artificial Intelligence, Machine Learning, Deep Learning, Large Language Models, RAG, Information Retrieval, Data Science, Programming, Web Development, Databases, Cloud Computing, Cybersecurity, Mathematics, Science, Business, Other, History.

The taxonomy is centralized so it can be changed without scattering label strings through the code.

## Dataset

`ml/data/starter_dataset.jsonl` contains 5,000 examples (4,940 unique normalized texts after deduplication) covering single-label, multi-label, overlapping, and domain-specific cases across all 17 topics.

Format:

```json
{"text":"...", "labels":["Artificial Intelligence", "RAG"]}
```

The training code uses leak-free stratified group splitting with Union-Find near-duplicate clustering to create a 70/15/15 train/validation/test split with 0 data leakage.

## Install

From `backend/`:

```bash
pip install -r requirements.txt
```

The ML additions require `torch`, `transformers`, `accelerate`, `scikit-learn`, and `sentencepiece`.

The base Transformer is downloaded from Hugging Face the first time training is run.

## Train

```bash
python -m ml.train
```

Configuration can be supplied through environment variables:

- `RECALL_CLASSIFIER_MODEL`
- `RECALL_CLASSIFIER_LR`
- `RECALL_CLASSIFIER_BATCH_SIZE`
- `RECALL_CLASSIFIER_EPOCHS`
- `RECALL_CLASSIFIER_MAX_LENGTH`
- `RECALL_CLASSIFIER_CHUNK_SIZE`
- `RECALL_CLASSIFIER_CHUNK_OVERLAP`
- `RECALL_CLASSIFIER_THRESHOLD`
- `RECALL_CLASSIFIER_SEED`
- `RECALL_CLASSIFIER_TOP_K_CHUNKS`

Example CPU development configuration:

```powershell
$env:RECALL_CLASSIFIER_BATCH_SIZE="4"
$env:RECALL_CLASSIFIER_EPOCHS="2"
$env:RECALL_CLASSIFIER_MAX_LENGTH="128"
python -m ml.train
```

## Evaluate

```bash
python -m ml.evaluate
```

The evaluation reports micro F1, macro F1, precision, recall, Hamming loss, and a per-label classification report. It also prints text → predictions/probabilities → actual labels.

No metrics are hard-coded into this repository.

## Inference

```bash
python -m ml.predict "A RAG system retrieves relevant documents before an LLM generates an answer."
```

Programmatic interface:

```python
from ml.classifier import TopicClassifier

classifier = TopicClassifier()
classifier.predict(text)
classifier.predict_article(long_article)
classifier.predict_tags(long_article)
```

`predict_article` chunks long text using a configurable word window and overlap. For each label it averages the top-K chunk probabilities. This is intentionally simple and inspectable; it prevents one irrelevant section from suppressing a topic that is strongly present elsewhere.

## Saved model

Training writes the reloadable model/tokenizer/config under:

`ml/models/recall-topic-classifier/`

Large model artifacts should not be committed to Git. The repository `.gitignore` ignores this trained-model directory while keeping the code and starter dataset.

## Webpage source neutrality

The classifier accepts text/chunks only. It does not receive a URL, domain, site
name, or a source-specific feature. A programming article from Wikipedia and a
programming article from another website are therefore the same type of ML
input.

Webpage extraction is handled outside this package by `app/webpage.py`. The
production ingestion flow creates a generic `WebDocument`, then passes its
chunks to `TopicClassifier` and the local embedding model.

## Model Evaluation

The classifier checkpoint was evaluated on the leak-free test split (741 samples) with exact metrics recorded in `ml/data/evaluation_report.json`:
- Subset Accuracy: 91.23%
- Hamming Accuracy: 99.31%
- Micro F1: 95.29%
- Macro F1: 94.92%
- Weighted F1: 95.32%
- Micro Precision: 94.73%
- Micro Recall: 95.86%
- Data Leakage: 0 overlap (PASS)
