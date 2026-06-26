# AMMORE

Agentic literature review built on top of [mmore](https://github.com/swiss-ai/mmore). Instead of a single retrieval pass, AMMORE uses a multi-agent loop to decompose the question, retrieve iteratively, check coverage, and write a structured, sourced synthesis.

Built with [AutoGen](https://github.com/microsoft/autogen).

## How it works

```
question
  └─> Planner      breaks it into a few sub-queries (you approve or edit them)
        └─> Retriever  searches mmore (and Tavily on fallback)
              └─> Critic   verdict: COVERAGE_OK or NEEDS_MORE (+ follow-up queries)
                    └─> Writer   structured synthesis, sourcing only the retrieved chunks
```

The user validates the sub-queries before the loop starts, useful for catching when the Planner misunderstands the question. The Critic is a prompted agent; a dspy structured-output variant lives in `critic.py` but is not wired into the loop, it made the follow-up queries drift off the original question (see the report).

## Setup

```bash
git clone <repo-url> AMMORE
cd AMMORE
uv venv --python 3.11
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS
uv pip install -e .
cp .env.example .env           # then add your API keys
```

This pulls mmore with the extras AMMORE needs (`process`, `index`, `rag`, `api`) and installs AMMORE in one go.

On Linux/macOS, AMMORE uses Milvus Lite automatically. On Windows, start Milvus first:

```powershell
docker compose -f docker-compose-milvus.yml up -d
```

The first corpus run downloads the embedding models and can take a few minutes. Later runs reuse the indexed corpus when the files did not change.

`.env` holds only the API keys:
```
OPENAI_API_KEY=your-key-here    # the system (gpt-4o-mini)
KIMI_API_KEY=your-key-here      # the evaluation judge (Moonshot)
TAVILY_API_KEY=your-key-here    # optional, for web fallback
# MISTRAL_API_KEY=...           # only if you set provider: mistral
```

`config.yaml` holds the things you actually tune. Defaults for everything else live in `ammore/config.py`.
```yaml
provider: openai            # openai | mistral | ollama

openai:
  model: gpt-4o-mini

retrieval:
  max_matches: 8
  min_similarity: -1

loop:
  context_mode: full        # full (large-context models) | headtail (small models)

websearch:
  enabled: false            # needs TAVILY_API_KEY

document_metadata:
  mode: llm                 # cheap | llm | none
```

## Usage

Point AMMORE at a folder of documents (or a pre-extracted `.json` collection) with `--corpus`. It indexes them once, then runs the agent loop:

```bash
python -m ammore "What are the main evaluation methods for multimodal LLMs?" --corpus path/to/papers/
```

Or without `--corpus` if you already have an mmore retriever running with an indexed collection:

```bash
python -m ammore "your question"
```

AMMORE auto-launches the mmore retriever as a subprocess. The synthesis is saved to `outputs/review_<timestamp>.md`.

## Evaluation

Reproduce the blind A/B comparison against the single-pass baseline. Both systems share the same Writer prompt and the same retrieval, so the score gap reflects the agent loop, not the prompt.

```bash
# 1) run both systems on the same questions (one question per line)
python -m ammore.eval questions/your_set.txt --corpus path/to/papers/ --out outputs/eval_answers.json
#    use --web-only instead of --corpus for the corpus-free web setting

# 2) score both answers blind with the judge
python -m ammore.judge --answers outputs/eval_answers.json --out outputs/eval_scores.json
#    add --domain humanitarian for medical corpora (adds the ethical_awareness criterion)
```

The judge (Kimi K2.6, Moonshot) is from a different provider than the system under test, which limits self-preference bias. It needs `OPENAI_API_KEY` (system) and `KIMI_API_KEY` (judge).

## Web UI

There is also a web interface. Two terminals:

```bash
# terminal 1: the API server (same venv as the CLI)
python -m ammore.server

# terminal 2: the frontend (needs Node 18+; npm install on first run)
cd ui
npm run dev
```

Then open http://localhost:3000. The flow mirrors the CLI: pick a corpus (path or file upload), ask a question, edit the generated sub-questions, watch the agents work, read the review. Past reviews from `outputs/` show up in the sidebar. The CLI keeps working independently.

## Web search

The Retriever falls back to Tavily when the corpus does not cover the question. Off by default: set `websearch.enabled: true` and add `TAVILY_API_KEY` to `.env` (free key at https://tavily.com, 1k queries/month on the student tier).

## Project structure

```
ammore/
  main.py               # entry point: plan, validate, run
  agents.py             # Planner, Retriever, Critic, Writer + search tools
  loop.py               # the SelectorGroupChat loop and routing
  critic.py             # dspy Critic variant (not wired in; see report)
  baseline.py           # single-pass baseline (shares the Writer prompt)
  eval.py               # blind A/B evaluation runner
  judge.py              # LLM-as-judge scoring (Kimi)
  corpus.py             # per-corpus indexing for --corpus
  document_metadata.py  # per-corpus title/abstract overview
  mmore_client.py       # calls the mmore retriever API
  web_search.py         # Tavily web search tool
  llm.py                # OpenAI / Mistral / Ollama client
  config.py             # config.yaml + .env loader
  retriever_launcher.py # launches the mmore retriever subprocess
  server.py             # FastAPI back end for the web UI
config.yaml             # user-facing config
ui/                     # Next.js + Chakra front end
```

## Development

```bash
pip install pre-commit ruff pyright
pre-commit install        # auto-runs ruff on every commit
ruff check . && ruff format .
pyright
```

## Design notes

- **Sub-questions:** 4-6 per question, generated by the Planner, shown to the user before the loop starts
- **Chunks:** `max_matches` per sub-query (default 8, set in `config.yaml`)
- **Critic:** a prompted agent returns `COVERAGE_OK` or `NEEDS_MORE` plus follow-up queries
- **Loop cap:** hard cap of 24 messages; the selector forces the Writer to run two messages before the cap
- **Models:** one model for all agents so the evaluation measures the loop, not per-agent differences. gpt-4o-mini for the reported runs; Mistral and Ollama are also supported

## What's not implemented yet

- Two-stage retrieval (document-level then chunk-level)
- Different models per agent role