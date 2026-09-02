import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from langgraph.graph import END, START, StateGraph

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

from tools import predict_risk, retrieve_guidance  # noqa: E402

load_dotenv()

# gemini-1.5-flash is retired. gemini-3.7-flash (the newest Flash-tier model)
# returned persistent 503 "high demand" errors against this API key; the API
# itself named gemini-3.6-flash as the replacement when an older model ID
# (gemini-2.5-flash) was requested with this key, and it responds reliably.
GEMINI_MODEL = "gemini-3.6-flash"

# Pulled from docs/model_card.md's "Known limitations" section so the LLM
# call below can't overclaim beyond what the underlying models actually
# support.
MODEL_CARD_CAVEATS = (
    "The risk model's dataset is small (4,463 rows, only 893 in the test set) and its "
    "~79% fatal rate reflects which incidents happened to get scraped with full "
    "narratives, not real-world incidence -- its output probabilities are not "
    "calibrated real-world risk. The data also ends in 2014, and depending on which "
    "path produced the estimate, it's either an untuned baseline model or a "
    "vocabulary-limited text model -- not a single validated risk system."
)

# Keyword sets and canonical phrasing mirror test_retrieval.py's TEST_QUERIES,
# which were already verified (Day 6 fix) to retrieve the correct top-3
# chunks for each category -- reusing that exact phrasing keeps retrieval
# quality the same instead of inventing new, unverified query wording per
# incident description.
HAZARD_CATEGORIES = [
    (("fall", "fell", "falling", "ladder", "roof", "height", "elevat"),
     "fall protection requirements for roofing work"),
    (("trench", "excavat", "cave-in", "cave in", "shoring", "sloping"),
     "trenching and excavation safety requirements"),
    (("electric", "electrocut", "voltage", "wire", "wiring", "shock", "generator", "power line"),
     "electrical hazard protection for construction workers"),
    (("scaffold", "scaffolding", "platform"),
     "scaffold safety requirements and load capacity"),
    (("struck", "falling object", "moving object", "run over", "crushed"),
     "struck-by hazards from falling or moving objects"),
]


def extract_guidance_query(description: str) -> str:
    """Builds a guidance-retrieval query from hazard-category keywords found
    in the description, rather than handing retrieve_guidance the raw
    narrative -- its embedding search is verified against short, topic-focused
    phrasing (see HAZARD_CATEGORIES), not arbitrary free text."""
    description_lower = description.lower()
    matched = [
        phrase for keywords, phrase in HAZARD_CATEGORIES
        if any(kw in description_lower for kw in keywords)
    ]
    if matched:
        return " ".join(matched)
    # No recognizable hazard category -- there's nothing better than a
    # generic query to hand retrieve_guidance here.
    return "general workplace safety hazard requirements"


class IncidentState(TypedDict, total=False):
    description: str
    risk: dict
    guidance_query: str
    guidance_text: str
    draft: str


def predict_risk_node(state: IncidentState) -> dict:
    risk = predict_risk.invoke({"description": state["description"]})
    return {"risk": risk}


def retrieve_guidance_node(state: IncidentState) -> dict:
    query = extract_guidance_query(state["description"])
    guidance_text = retrieve_guidance.invoke({"query": query})
    return {"guidance_query": query, "guidance_text": guidance_text}


DRAFT_SYSTEM_PROMPT = f"""You are drafting a safety recommendation for a human safety officer to review, based on a construction-incident risk model's output and retrieved OSHA guidance excerpts.

Hard requirements:
- This is a DRAFT for a human safety officer to review and act on -- never say or imply it should be auto-implemented or treated as a final determination.
- Explicitly cite which guidance document(s) you're drawing from, by the filename given after "source=" in each retrieved chunk.
- If the risk model output has low_confidence set to true (or a "caveat" field), explicitly say the risk estimate is uncertain due to limited input detail -- do not present the probability as a confident number in that case.
- Never make an absolute predictive claim about what will happen (e.g. "this incident will result in death" or "this will cause a fatality"). Speak only in terms of estimated/elevated/relative risk. {MODEL_CARD_CAVEATS}
- Plain language, addressed to a safety officer -- not technical jargon.
- 3-5 sentences."""

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set -- add it to a .env file (see .env.example) "
                "or export it in the environment."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def draft_recommendation_node(state: IncidentState) -> dict:
    client = _get_gemini_client()

    user_content = (
        f"Incident description: {state['description']}\n\n"
        f"Risk model output: {state['risk']}\n\n"
        f"Retrieved OSHA guidance:\n{state['guidance_text']}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_content,
        config=genai_types.GenerateContentConfig(system_instruction=DRAFT_SYSTEM_PROMPT),
    )
    return {"draft": response.text}


def build_graph():
    builder = StateGraph(IncidentState)
    builder.add_node("predict_risk", predict_risk_node)
    builder.add_node("retrieve_guidance", retrieve_guidance_node)
    builder.add_node("draft_recommendation", draft_recommendation_node)
    builder.add_edge(START, "predict_risk")
    builder.add_edge("predict_risk", "retrieve_guidance")
    builder.add_edge("retrieve_guidance", "draft_recommendation")
    builder.add_edge("draft_recommendation", END)
    return builder.compile()


GRAPH = build_graph()

TEST_CASES = [
    "An electrician fell 15 feet from a ladder while working on wiring in an unfinished building.",
    "Employee reported feeling dizzy after lunch and went home early for the day.",
    "A laborer was in a 10-foot trench installing a sewer line when the trench wall collapsed, partially burying him.",
]

if __name__ == "__main__":
    for description in TEST_CASES:
        print("=" * 80)
        print(f"Incident: {description}")
        print("=" * 80)

        result = GRAPH.invoke({"description": description})

        print("\n-- predict_risk --")
        print(result["risk"])

        print("\n-- retrieve_guidance --")
        print(f"Query used: {result['guidance_query']}")
        print(result["guidance_text"])

        print("\n-- draft_recommendation --")
        print(result["draft"])
        print()
