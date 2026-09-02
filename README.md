# EHS Intelligence: OSHA Construction Fatality Risk Platform

A construction-safety risk platform built on OSHA's historical accident
narratives. It predicts whether a described incident is associated with a
fatality, retrieves the OSHA compliance guidance most relevant to that
incident, and drafts a plain-language safety recommendation for a human
safety officer to review — it does not auto-implement anything.

Built incrementally (see commit history for the day-by-day progression from
data cleaning through the agent pipeline below).

## Architecture

The platform layers six pieces, from the ground up:

1. **Structured risk model (Day 4)** — `src/train_model.py` trains a Random
   Forest on structured features only (occupation, industry, fall
   involvement/distance, narrative length, cyclical month) — no NLP. Chosen
   over Logistic Regression for better balanced recall. Saved to
   `models/day4_risk_rf.joblib`.
2. **Text-only model (Day 5)** — `src/train_text_model.py` trains a TF-IDF +
   Logistic Regression classifier on the incident narrative text alone, with
   a custom analyzer (`src/text_analyzer.py`) that filters out ID-numbering
   and calendar-token artifacts. Saved to `models/day5_text_logreg.joblib`.
3. **Combined model (Day 5b)** — `src/train_combined_model.py` concatenates
   TF-IDF text features with the structured features from step 1 into a
   single Logistic Regression. Evaluated for comparison; not persisted to
   disk (it's a comparison baseline, not something downstream code loads).
4. **RAG knowledge base (Day 6)** — `src/build_rag_index.py` chunks OSHA
   compliance documents (`data/compliance_docs/`) and indexes them into a
   persistent Chroma collection (`data/chroma_db/`), with generic
   legal-boilerplate chunks flagged so they don't out-rank topic-specific
   guidance at retrieval time (`src/test_retrieval.py`).
5. **LangGraph agent (Day 7)** — `src/agent/tools.py` wraps `predict_risk`
   (routes free text to the Day 4 structured model when occupation/industry
   can be confidently extracted, otherwise falls back to the Day 5 text
   model, flagging `low_confidence` on sparse input) and `retrieve_guidance`
   as LangChain tools. `src/agent/graph.py` chains them into a linear graph
   — `predict_risk` → `retrieve_guidance` → `draft_recommendation` — where
   the last step calls the Gemini API to draft a recommendation that always
   cites its guidance source(s) by filename, hedges explicitly when the risk
   estimate is low-confidence, and never claims to be more than a draft for
   human review.
6. **Streamlit dashboard (Day 9)** — `dashboard/app.py` puts both of the
   above in front of a user: an analytics tab (filterable incident table,
   industry/time-trend/occupation×injury Plotly charts, all reused from the
   Day 2 EDA notebook) and a chat tab that runs the same compiled LangGraph
   agent from Day 7 against a typed-in incident description, with the risk
   score, confidence flag, retrieved sources, and drafted recommendation all
   rendered live.

## Engineering highlights

A few things in here are worth a closer look than a feature-list bullet:

- **Day 5's TF-IDF artifact hunt.** The text model's early top FATAL
  coefficients were dominated by things like `2007`, `october`, and
  `employee employee` — calendar tokens and ID-numbering duplication
  artifacts from the narrative text, not real risk signal (full trail in
  `docs/model_card.md`'s Feature Cleaning Iteration History). Root-caused via
  direct coefficient inspection, not assumption, and the fix
  (`src/text_analyzer.py`'s custom analyzer) is shared by both the text-only
  and combined models rather than duplicated.
- **Day 9's `stream()` over `invoke()` choice.** The chat tab runs
  `GRAPH.stream(..., stream_mode="updates")` instead of the more obvious
  `GRAPH.invoke(...)`. The difference matters concretely: if the last node
  (`draft_recommendation`, a live Gemini call) fails — e.g. a missing
  `GOOGLE_API_KEY` — `invoke()` would discard everything the graph had
  already computed. `stream()` yields each node's output as it completes, so
  the risk score and retrieved guidance from the two earlier nodes still
  render even when the LLM call fails, with a clear inline error in place of
  the missing draft instead of a blank screen or a stack trace.

## Known limitations

See `docs/model_card.md` for full detail (training data, evaluation metrics,
feature-cleaning history). In brief:

- The ~79% fatality rate in the training data is a scrape-selection artifact
  of this specific dataset, not a real construction-industry statistic.
- Data spans 1984–2014 with uneven year coverage.
- No hyperparameter tuning was performed on any of the three models.
- The RAG corpus is a small, curated set of 9 documents — not comprehensive
  OSHA coverage.
- The drafted recommendations come from a general-purpose LLM (Gemini)
  reading the retrieved guidance and model output — they are not verified
  for correctness and can still misstate or omit safety-critical detail,
  which is why every draft is explicitly framed as a starting point for a
  human safety officer, never a final determination.
- This is a portfolio/demo project, not validated for real safety decisions.

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file at the project root (copy `.env.example`) with a Gemini
API key, needed only for `src/agent/graph.py`'s `draft_recommendation` step:

```
GOOGLE_API_KEY=your-key-from-https://aistudio.google.com/apikey
```

**No download step is needed.** The raw dataset
(`data/raw/OSHA_Acc-master/*.xlsx`) and the OSHA compliance PDFs/text used by
the RAG index (`data/compliance_docs/`) both ship pre-committed in this repo.

## Running the pipeline

Run in this exact order from the project root:

1. `python src/load_data.py` — cleans the raw OSHA Excel export into
   `data/processed/osha_clean.csv`.
2. `python src/build_features.py` — builds the structured feature matrix
   (`data/processed/features.csv`) that steps 3–5 all depend on.
3. `python src/train_model.py` — trains and saves the Day 4 structured Random
   Forest (`models/day4_risk_rf.joblib`), plus a SHAP summary plot.
4. `python src/train_text_model.py` — trains and saves the Day 5 text-only
   model (`models/day5_text_logreg.joblib`).
5. `python src/train_combined_model.py` — trains and evaluates the combined
   text + structured model (comparison only, nothing saved to disk).
6. `python src/build_rag_index.py` — chunks and embeds
   `data/compliance_docs/` into the Chroma vector store at
   `data/chroma_db/`.
7. `python src/agent/graph.py` — runs the full agent pipeline
   (`predict_risk` → `retrieve_guidance` → `draft_recommendation`) against
   three example incident descriptions. Requires `GOOGLE_API_KEY` to be set;
   without it, the first two steps still run and it fails cleanly at the
   Gemini call.

Steps 3, 4, and 5 each read `data/processed/features.csv` and/or
`data/processed/osha_clean.csv` produced by step 2 — step 2 must be re-run
first if the raw data or feature-engineering logic changes. Step 6 is
independent of steps 1–5 (it only touches `data/compliance_docs/`), and step
7 depends on the model artifacts from steps 3–4 and the index from step 6.

## Running the dashboard

Once steps 1–6 above have produced `models/` and `data/chroma_db/` (or you've
pulled them from the repo as committed), launch the dashboard:

```
streamlit run dashboard/app.py
```

The analytics tab needs nothing beyond `data/processed/osha_clean.csv`
(step 1). The chat tab needs the Day 4/5 models and the Chroma index (steps
3, 4, 6) plus `GOOGLE_API_KEY` — without a key, risk prediction and guidance
retrieval still work in the chat tab, and it shows a clear inline error in
place of the drafted recommendation rather than crashing.

## Project layout

```
src/
  load_data.py, sic_codes.py        Day 1: raw data cleaning
  build_features.py                 Day 3: structured feature engineering
  train_model.py                    Day 4: structured Random Forest
  train_text_model.py               Day 5: TF-IDF text-only model
  text_analyzer.py                  shared picklable TF-IDF analyzer
  train_combined_model.py           Day 5b: combined model comparison
  build_rag_index.py                Day 6: RAG index build
  test_retrieval.py                 Day 6: retrieval + boilerplate re-ranking
  agent/tools.py                    Day 7: predict_risk, retrieve_guidance
  agent/graph.py                    Day 7: LangGraph pipeline + Gemini draft
dashboard/app.py                    Day 9: Streamlit analytics + chat UI
data/
  raw/                               pre-committed source Excel files
  processed/                         cleaned data + feature matrix
  compliance_docs/                   OSHA guidance source docs for RAG
  chroma_db/                         persistent Chroma vector store
models/                              saved Day 4 / Day 5 model artifacts
docs/model_card.md                   full model documentation and limitations
notebooks/01_eda.ipynb               exploratory data analysis
.env.example                        template for the GOOGLE_API_KEY secret
```
