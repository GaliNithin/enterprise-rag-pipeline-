# Enterprise RAG Pipeline

A production-ready Retrieval-Augmented Generation pipeline with advanced retrieval strategies, hallucination guardrails, and evaluation tooling. Built for regulated enterprise environments (insurance, healthcare, finance).

## Architecture

```
Documents → Chunking Strategy → Embeddings → Vector Store
                                                    ↓
User Query → Query Rewriting → Retrieval (Parent-Doc) → Reranking → LLM → Response
                                                                        ↓
                                                              Guardrail Check
```

## Features

- **Parent-Document Retrieval** — indexes small chunks for precision, retrieves full parent docs for context
- **Query Rewriting** — rewrites ambiguous queries before retrieval to improve recall
- **Hybrid Search** — combines dense (vector) + sparse (BM25) retrieval
- **Hallucination Guardrails** — scores response groundedness against retrieved context
- **RAGAS Evaluation** — built-in evaluation with faithfulness, answer relevancy, context precision metrics

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI / Azure OpenAI key
python ingest.py --docs ./sample_docs
python query.py --question "What is the deductible for commercial auto?"
```

## Project structure

```
enterprise-rag-pipeline/
├── ingest.py            # Document ingestion and chunking
├── retriever.py         # Parent-doc retrieval + hybrid search
├── rewriter.py          # Query rewriting with LLM
├── guardrails.py        # Hallucination detection
├── evaluate.py          # RAGAS evaluation pipeline
├── pipeline.py          # End-to-end orchestration
├── notebooks/
│   └── rag_walkthrough.ipynb
├── sample_docs/         # Sample insurance policy documents
├── requirements.txt
└── .env.example
```

## Stack

`LangChain` `OpenAI / Azure OpenAI` `Pinecone` `ChromaDB` `RAGAS` `FastAPI` `Python 3.11`

## Evaluation results (sample docs)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.91 |
| Answer relevancy | 0.87 |
| Context precision | 0.83 |
| Context recall | 0.79 |
