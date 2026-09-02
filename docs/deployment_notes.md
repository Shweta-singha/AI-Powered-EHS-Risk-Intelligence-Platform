# Deployment notes (Streamlit Community Cloud / Render)

Research findings from a pre-deployment review — nothing has been deployed
yet. Numbers below are measured locally in this repo's environment, not
guessed.

## RAG index: ship committed, don't rebuild on deploy

`data/chroma_db/` is 4.7 MB and already git-tracked. It's deterministic from
the 9 source docs in `data/compliance_docs/`, so rebuilding it during deploy
would only add a `sentence-transformers` import (500+ MB, see below) to the
build step for zero benefit. Ship it as-is.

**Risk found and fixed**: `requirements.txt` had `chromadb>=1.0` (floating
lower bound). If a newer `chromadb` resolved at deploy time uses a different
sqlite schema than the one that built the committed index, retrieval could
break at runtime. Fixed by pinning `chromadb==1.5.9` — the exact version that
built the committed index in this repo.

## Secrets: both platforms work with zero code changes

`src/agent/graph.py` reads the Gemini key via plain `os.environ.get("GOOGLE_API_KEY")`
after `load_dotenv()`. Both platforms inject secrets as real process
environment variables, so no code change is needed for either:

- **Streamlit Community Cloud**: paste into the app's Secrets dialog
  (`secrets.toml` format) at deploy time or later in app settings; exposed
  via both `st.secrets` and as env vars.
- **Render**: set as a plain Environment Variable in the service dashboard
  or `render.yaml`.

## Memory: the real open risk

Measured actual RSS growth locally (not just on-disk package size) while
loading everything the agent path needs:

| Stage | RSS |
|---|---|
| Bare Python process | 18 MB |
| `import torch` | 211 MB |
| `+ import chromadb` | 267 MB |
| `+ import sentence_transformers` | 526 MB |
| `+ load embedding model & Chroma collection` | 556 MB |
| `+ load Day 4 & Day 5 joblib models` | 616 MB |
| `+ one predict_risk + retrieve_guidance call` | **704 MB** |

That excludes Streamlit's own process overhead and the dashboard tab's
`pandas`/`plotly` usage in the same process.

Current published free-tier limits (verified via search, since these change
and shouldn't be assumed from memory):

- **Render free tier: 512 MB RAM.** Just `import torch` +
  `import sentence_transformers` (526 MB) already exceeds this before a
  model even loads. **Very unlikely to run at all on Render's free tier as
  currently built**, without removing `torch`/`sentence-transformers` from
  the request path entirely.
- **Streamlit Community Cloud free tier: 1 GB RAM.** 704 MB leaves ~300 MB
  of headroom for Streamlit itself plus the dashboard tab's libraries —
  plausible but genuinely tight; real OOM risk under normal use, not just
  edge cases. Worth a real deploy test before trusting it, since
  Streamlit's own overhead and concurrent-session behavior aren't fully
  reproducible locally.

Also relevant: Render's free tier spins down after 15 min idle (30-60s cold
container start) on top of this app's own ~9s model-load cold start;
Streamlit Community Cloud only sleeps after 12 hours idle.

## Investigated: ONNX backend / smaller embedding model — concluded not worth pursuing

Tested directly rather than assumed. Two things ruled this out:

1. **`sentence-transformers` imports `torch` unconditionally at package-import
   time**, regardless of which backend you later choose for inference.
   Measured: `torch` was already in `sys.modules` and RSS was already at
   478 MB immediately after `from sentence_transformers import SentenceTransformer`
   — before specifying `backend="onnx"` or loading any model. `pip show
   sentence-transformers` also lists `torch` as a hard `Requires`, not an
   optional extra. Choosing `backend="onnx"` only changes what runs
   inference after the model loads, not what gets imported to get there.
2. Actually using the ONNX backend requires installing `optimum` and ONNX
   Runtime extras (`pip install sentence-transformers[onnx]`) — a new
   dependency, which fails the "low-effort" bar on its own even before
   considering point 1.
3. A smaller embedding model wouldn't meaningfully help either: the ~526 MB
   baseline is dominated by importing `torch`+`transformers` as libraries,
   not by `all-MiniLM-L6-v2`'s own weights (loading the actual model only
   added ~30 MB on top of the import cost). A smaller model would shave a
   few MB off that 30 MB, not the 500+ MB import tax.

The only way to meaningfully cut this would be dropping
`sentence-transformers`/`torch` from the request path entirely (a pure ONNX
Runtime pipeline with no `sentence-transformers` dependency, or a hosted
embedding API), which would require re-embedding `data/chroma_db/` and
rewriting `src/test_retrieval.py`'s model loading — a real architecture
change, not a config tweak. No changes made; decision was to proceed to
deploy-and-monitor instead.
