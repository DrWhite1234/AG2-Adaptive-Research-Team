# AG2 Adaptive Research Team

A multi-agent research assistant built on **AG2 (AutoGen 2)** and **HuggingFace Inference Providers**. Upload documents, ask a research question, and a coordinated team of five specialised AI agents will triage, research, verify, and synthesise a cited answer — automatically deciding whether to draw from your local files or the live web.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [File Structure](#file-structure)
- [Models](#models)
- [Packages & Dependencies](#packages--dependencies)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [How to Use](#how-to-use)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

The AG2 Adaptive Research Team is a **Streamlit web application** that demonstrates agentic AI workflows. Rather than sending a question to a single LLM, the app orchestrates a pipeline of five purpose-built agents that collaborate to produce a well-evidenced, verified answer.

Key capabilities:

- **Automatic routing** — a triage agent decides whether local documents or a web search is the better source
- **Local document Q&A** — upload PDFs, `.txt`, or `.md` files; the agent searches them with TF-IDF-style keyword matching
- **Web fallback** — if local documents are insufficient or absent, a web research agent queries a public SearxNG instance
- **Verification pass** — a dedicated verifier agent checks whether the evidence is sufficient before the final answer is written
- **Cited synthesis** — the synthesiser agent produces a final answer with explicit references to the evidence sources

---

## Architecture

```
User Question
      │
      ▼
┌─────────────┐
│ Triage Agent│  ──► Decides: "local" or "web"
└─────────────┘
      │
  ┌───┴────────────────────┐
  │                        │
  ▼                        ▼
┌──────────────┐    ┌─────────────────┐
│ Local        │    │ Web Research    │
│ Research     │    │ Agent           │
│ Agent        │    │ (SearxNG)       │
└──────┬───────┘    └────────┬────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
         ┌────────────────┐
         │ Verifier Agent │  ──► Checks evidence gaps
         └───────┬────────┘
                 ▼
         ┌───────────────────┐
         │ Synthesiser Agent │  ──► Final cited answer
         └───────────────────┘
```

All five agents share the same LLM backend (configurable) but carry distinct system prompts that constrain their behaviour and output format.

---

## Agent Pipeline

### 1. Triage Agent (`triage_agent`)
Receives the user's question and a summary of any uploaded documents. Returns a JSON object specifying:
- `route`: `"local"` or `"web"`
- `confidence`: a 0–1 score
- `rationale`: a short explanation

The route drives every subsequent step. If no local documents were uploaded and web fallback is enabled, the route is forced to `"web"` regardless of the triage agent's output.

### 2a. Local Research Agent (`local_research_agent`)
Activated when the route is `"local"`. Receives the top-5 document chunks most relevant to the question (retrieved via keyword overlap scoring) and returns JSON containing:
- `evidence`: a list of `{source, summary}` objects
- `draft_answer`: a prose answer grounded in the excerpts

### 2b. Web Research Agent (`web_research_agent`)
Activated when the route is `"web"`. Receives the top-5 results from a SearxNG search and returns the same `evidence` / `draft_answer` JSON structure.

### 3. Verifier Agent (`verifier_agent`)
Receives the draft answer and the evidence list. Returns:
- `verdict`: `"sufficient"` or `"insufficient"`
- `gaps`: a list of identified knowledge gaps

The verifier result is passed to the synthesiser to inform how confidently the final answer should be stated.

### 4. Synthesiser Agent (`synthesizer_agent`)
Receives the question, draft answer, evidence, and verifier verdict. Produces the final free-text answer with inline citations to the evidence sources.

---

## File Structure

```
project/
│
├── app.py              # Streamlit UI — entry point
├── agents.py           # Agent definitions and LLM configuration
├── router.py           # Pipeline orchestration logic
├── tools.py            # Document loading, chunking, search, and SearxNG wrapper
└── requirements.txt    # Python dependencies
```

### `app.py`
The Streamlit front end. Handles:
- API key and model input (sidebar)
- File upload widget
- Calling `run_pipeline()` from `router.py`
- Displaying triage JSON, evidence JSON, verifier JSON, and the final answer

### `agents.py`
Defines `build_agents()`, which constructs five `AssistantAgent` instances with individual system prompts, all sharing the same `llm_config`. Also exposes `make_llm_config()` and the convenience wrapper `run_agent()`.

Key constants:
| Constant | Value | Purpose |
|---|---|---|
| `HF_BASE_URL` | `https://router.huggingface.co/v1` | HuggingFace Inference Providers router |
| `DEFAULT_MODEL` | `Qwen/Qwen2.5-72B-Instruct:novita` | Default model + provider |

### `router.py`
Contains `run_pipeline()`, which wires the agents together sequentially. Also includes `_extract_json()` (regex-based JSON extraction from LLM output) and `_summarize_chunks()` (formats retrieved chunks for the prompt).

### `tools.py`
Four responsibilities:
1. **`load_documents()`** — reads uploaded PDF and text files into `Document` objects
2. **`chunk_text()`** — splits document text into overlapping 800-word chunks (120-word overlap)
3. **`search_local()`** — keyword-overlap scoring to retrieve the top-k most relevant chunks
4. **`run_searxng()`** — wraps `SearxngSearchTool` from AG2 to query the public SearxNG instance

---

## Models

### Default Model
**`Qwen/Qwen2.5-72B-Instruct`** served via the **Novita** inference provider.

Qwen 2.5 72B Instruct is a 72-billion parameter open-weight instruction-tuned model from Alibaba Cloud. It consistently ranks among the top open-weight models on reasoning, instruction-following, and coding benchmarks and is available free-tier through HuggingFace Inference Providers.

### HuggingFace Inference Providers
The app uses HuggingFace's unified Inference Providers router (`https://router.huggingface.co/v1`), which is OpenAI API-compatible. You select a backend provider by appending a suffix to the model name.

| Suffix | Provider | Notes |
|---|---|---|
| `:novita` | Novita AI | Default; good throughput for Qwen models |
| `:auto` | Auto-selected | HF picks the fastest available provider |
| `:together` | Together AI | Strong alternative for open-weight models |
| `:sambanova` | SambaNova | Fast inference; supports Llama and Qwen |
| `:fireworks-ai` | Fireworks AI | Low-latency option |

**Model format:** `<org>/<model-name>:<provider>`

Examples:
```
Qwen/Qwen2.5-72B-Instruct:novita
Qwen/Qwen2.5-72B-Instruct:auto
meta-llama/Llama-3.3-70B-Instruct:sambanova
mistralai/Mixtral-8x7B-Instruct-v0.1:together
```

Browse all models and their supported providers at [huggingface.co/models](https://huggingface.co/models) using the **Inference Providers** filter.

---

## Packages & Dependencies

### `requirements.txt`
```
ag2[openai]>=0.11.0
streamlit>=1.33.0
pypdf>=4.2.0
openai>=1.30.0
```

### Package Details

#### `ag2[openai]` — AutoGen 2
The core multi-agent framework. AG2 (formerly AutoGen) provides `AssistantAgent`, the `generate_reply()` interface, and the `SearxngSearchTool`. The `[openai]` extra installs the OpenAI-compatible client used to communicate with HuggingFace's router.

- GitHub: [github.com/ag2ai/ag2](https://github.com/ag2ai/ag2)
- Docs: [docs.ag2.ai](https://docs.ag2.ai)

#### `streamlit` — Web UI Framework
Powers the interactive browser interface — file upload, sidebar configuration, and results display. No HTML or JavaScript required.

- Docs: [docs.streamlit.io](https://docs.streamlit.io)

#### `pypdf` — PDF Text Extraction
Extracts plain text from each page of uploaded PDF files. Used in `load_documents()` inside `tools.py`.

- Docs: [pypdf.readthedocs.io](https://pypdf.readthedocs.io)

#### `openai` — OpenAI Python SDK
Pulled in transitively by `ag2[openai]`. The HuggingFace Inference Providers router exposes an OpenAI-compatible REST API, so the standard `openai` client handles all LLM calls without any custom HTTP code.

- Docs: [platform.openai.com/docs/libraries](https://platform.openai.com/docs/libraries)

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- A free [HuggingFace account](https://huggingface.co) and access token

### Steps

**1. Clone or download the project**
```bash
git clone <your-repo-url>
cd ag2-adaptive-research-team
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Get a HuggingFace access token**
1. Sign in at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens**
3. Create a token with **Read** scope (and optionally **Make calls to Inference Providers** scope)
4. Copy the token — it starts with `hf_`

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## How to Use

1. **Enter your HuggingFace token** in the sidebar
2. **Optionally change the model** — keep the `:<provider>` suffix (e.g. `:novita`)
3. **Toggle Web Fallback** on or off depending on your needs
4. **Upload documents** (PDF, TXT, or MD) under section 1 — optional; if none are uploaded the app routes straight to web search
5. **Type your research question** under section 2
6. **Click "Run Research"** and wait for the pipeline to complete
7. Review the four result panels:
   - **Routing Decision** — the triage agent's JSON verdict
   - **Evidence** — source summaries collected by the research agent
   - **Verifier** — sufficiency verdict and identified gaps
   - **Final Answer** — the synthesised, cited response

---

## Configuration Reference

| Setting | Location | Default | Description |
|---|---|---|---|
| HuggingFace token | Sidebar / env var `HF_TOKEN` | — | Required for all LLM calls |
| Model | Sidebar | `Qwen/Qwen2.5-72B-Instruct:novita` | Any HF Inference Providers model with `:provider` suffix |
| Web fallback | Sidebar toggle | `True` | Enables SearxNG web search when route is "web" |
| SearxNG URL | `app.py` constant | `https://searxng.site/search` | Public SearxNG instance; replace with self-hosted URL if rate-limited |
| Chunk size | `tools.py` | `800` words | Controls document chunk size for local indexing |
| Chunk overlap | `tools.py` | `120` words | Overlap between consecutive chunks to preserve context |
| Top-k chunks | `router.py` | `5` | Number of document chunks passed to the local research agent |
| Max web results | `router.py` | `5` | Number of SearxNG results passed to the web research agent |
| LLM temperature | `agents.py` | `0.2` | Lower = more deterministic JSON output from agents |

---

## Troubleshooting

**`Cannot POST /v1/chat/completions`**
The base URL is wrong. Ensure `HF_BASE_URL` in `agents.py` is `https://router.huggingface.co/v1`.

**`Cannot POST /models/.../v1/chat/completions`**
You are using the old per-model URL format. Switch to the router URL above.

**Model returns 404 or "model not found"**
The model name is missing the provider suffix. Add `:novita`, `:auto`, or another supported provider. Example: `Qwen/Qwen2.5-72B-Instruct:novita`.

**SearxNG returns no results or times out**
The public instance at `searxng.site` may be rate-limiting your IP. Either wait and retry, or host your own SearxNG instance and update `SEARXNG_BASE_URL` in `app.py`.

**Agent returns empty or malformed JSON**
Increase `temperature` slightly (e.g. `0.3`) or add more explicit JSON format instructions in the relevant agent's `system_message`. The `_extract_json()` helper in `router.py` is tolerant of surrounding text, but completely empty responses will produce empty dicts.

**`pypdf` fails to extract text from a PDF**
Some PDFs are scanned images with no embedded text layer. OCR is not included — consider pre-processing with a tool like `ocrmypdf` before uploading.
