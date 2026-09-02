# Resume bullets

Drafted from verified, committed results in this repo (see `docs/model_card.md`
and the Day 6/7/9 commits). Non-fatal recall 51%→78% and ROC-AUC 0.79→0.85 are
confirmed against the final, post-bigram-fix numbers in `docs/model_card.md`
(explicitly marked there as "the number to cite going forward"), not an
intermediate iteration.

- Built and evaluated three fatality-risk classifiers (Random Forest, TF-IDF
  Logistic Regression, and a combined text+structured model) in scikit-learn
  on 4,463 OSHA construction incident records, improving non-fatal-case
  recall from 51% to 78% (ROC-AUC 0.79→0.85) by fusing narrative text with
  structured features.

- Root-caused spurious signal in a TF-IDF fatality classifier by tracing its
  top coefficients to calendar-token and duplicate-bigram artifacts (e.g.,
  "employee employee") rather than genuine risk language, built a custom
  tokenizer to filter them, and separately confirmed explicit forensic
  vocabulary ("coroner," "autopsy") was not a contributing leakage source.

- Built a RAG retrieval system (ChromaDB, sentence-transformers) indexing 152
  chunks from 9 OSHA compliance documents, then diagnosed and fixed a ranking
  bug where generic legal boilerplate was outranking topic-specific guidance
  — verified corrected top-3 retrieval across 5 hazard categories.

- Designed a LangGraph agent pipeline (predict_risk → retrieve_guidance →
  draft_recommendation) that routes incident text to the appropriate risk
  model, flags low-confidence predictions on sparse input, and drafts
  Gemini-generated, source-cited safety recommendations; exposed via a
  Streamlit dashboard where resource caching cut repeat-query latency by
  ~10 seconds.
