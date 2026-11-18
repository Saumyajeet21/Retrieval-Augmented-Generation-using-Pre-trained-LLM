# Retrieval-Augmented Generation (RAG) Using a Pre-trained LLM

**Practical No. 06 — Generative AI**  
**Project:** Aster Robotics — Internal Operations Knowledge Assistant  
**Author:** Saumyajeet Singh Bhati

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fx9g6RXGQe40V1zhWqShreJhwrglV-Ll?usp=sharing)

## Overview

This Google Colab project demonstrates **Retrieval-Augmented Generation (RAG)** for answering questions about a fictional company's internal policies. It retrieves relevant document passages from a vector database and supplies them to a pre-trained language model, then compares the results with answers from the **same LLM without retrieval**.

> **Note:** Aster Robotics and its policies are fictional sample data for educational purposes. The notebook includes an evaluation routine; scores should be reported only after executing it.

## Objectives

- Load and clean domain-specific `.txt` and `.pdf` documents.
- Split text into overlapping chunks and generate semantic embeddings.
- Index document chunks in ChromaDB and retrieve relevant evidence.
- Generate answers with a pre-trained LLM using retrieved context.
- Compare RAG answers with LLM-only answers and inspect potential hallucinations.

## Architecture

```text
TXT / PDF policy documents
          |
     Text cleaning
          |
 Chunking (480 characters, 90 overlap)
          |
 all-MiniLM-L6-v2 embeddings
          |
     ChromaDB index
          |
Question -> Query embedding -> Top-3 retrieval
                                   |
                         Context + question
                                   |
                          FLAN-T5 Base
                                   |
                           RAG answer

Question --------------------> FLAN-T5 Base
                                (no context)
                                   |
                            LLM-only answer
                                   |
                        Side-by-side evaluation
```

## Technology Stack

| Component | Technology |
| --- | --- |
| Programming language | Python |
| Environment | Google Colab / Jupyter Notebook |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Pre-trained LLM | `google/flan-t5-base` |
| Vector database | ChromaDB |
| PDF parsing | `pypdf` |
| Other libraries | `transformers`, `torch`, `sentence-transformers`, `pandas`, `numpy` |

## Dataset

The notebook creates five sample `.txt` policy documents in `/content/aster_docs`:

| File | Subject |
| --- | --- |
| `hr_policy.txt` | Human resources, leave and probation |
| `it_security.txt` | Passwords and security incident reporting |
| `engineering.txt` | Engineering workflow, reviews and sprints |
| `facilities.txt` | Workplace facilities and emergency procedures |
| `learning.txt` | Employee learning and training benefits |

The document loader also supports adding `.txt` or `.pdf` files to the same folder.

## How to Run

1. Open the [Google Colab notebook](https://colab.research.google.com/drive/1fx9g6RXGQe40V1zhWqShreJhwrglV-Ll?usp=sharing), or upload `Aster_Robotics_RAG_Practical_06.ipynb` to Colab.
2. Optionally select **Runtime → Change runtime type → T4 GPU** for faster model inference (CPU is also supported).
3. Select **Runtime → Run all**. The notebook installs dependencies and generates its sample dataset.
4. Inspect the document chunks, retrieval results and RAG vs LLM-only comparison.
5. In the interactive demonstration cell, enter a question about the sample company policies.

The notebook uses Hugging Face model downloads, so an internet connection is needed on the first run. No paid LLM API key is required.

## Example Questions

- How many paid annual leave days do employees receive?
- Within how many minutes must a lost laptop be reported?
- How many independent code reviewers are required for a release?
- Where is the emergency assembly location?
- What is the yearly employee training allowance?
- How long is the engineering sprint cycle?

**Out-of-scope test:** “Who is the CEO of Aster Robotics?” The sample documents do not specify a CEO; the RAG prompt instructs the model to respond **“Not found in the documents”** when the context does not contain an answer. Actual model behavior should be checked during execution.

## Evaluation

The notebook evaluates six domain-specific questions with expected answers. It displays:

- **RAG response** and **LLM-only response** for each question.
- **Keyword-match accuracy:** whether the response contains an expected answer keyword (a heuristic, not full factual verification).
- **Retrieval source hit rate:** whether the expected source document appears among the retrieved passages.
- **Source attribution:** source filenames associated with retrieved chunks.

The notebook computes the metrics during execution; no fixed accuracy scores are claimed here. Review generated responses manually for correctness, grounding and hallucinations.

## Limitations

- Character-based chunking can split a sentence or fact across boundaries.
- Top-3 semantic retrieval can miss relevant passages.
- FLAN-T5 Base may ignore context or generate incomplete answers.
- Keyword matching can misclassify semantically correct or incorrect answers.
- Source filenames identify retrieved evidence; they do not guarantee the answer is faithful to that evidence.

Possible improvements include hybrid retrieval, reranking, larger instruction-tuned models, semantic evaluation and a Streamlit or Gradio interface.

## Repository and Notebook

- **GitHub:** https://github.com/Saumyajeet21/Retrieval-Augmented-Generation-using-Pre-trained-LLM
- **Google Colab:** https://colab.research.google.com/drive/1fx9g6RXGQe40V1zhWqShreJhwrglV-Ll?usp=sharing

## Conclusion

This practical implements an end-to-end RAG workflow using document preprocessing, overlapping chunks, semantic embeddings, ChromaDB retrieval and FLAN-T5 Base answer generation. Comparing grounded answers with LLM-only responses demonstrates how retrieval can help with domain-specific knowledge while also highlighting retrieval and generation limitations.
