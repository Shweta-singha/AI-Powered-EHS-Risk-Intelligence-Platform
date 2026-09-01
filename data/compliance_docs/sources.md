# Compliance Document Sources

All documents are real, public OSHA.gov material, downloaded 2026-09-01. Local
filenames match the names referenced by `src/build_rag_index.py` and
`src/test_retrieval.py`.

| File | Risk category | Source |
|---|---|---|
| `1926.501_fall_protection_duty.txt` | Falls | 29 CFR 1926.501 — Duty to have fall protection. <https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.501> |
| `1926.502_fall_protection_systems.txt` | Falls | 29 CFR 1926.502 — Fall protection systems criteria and practices. <https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.502> |
| `1926.416_electrical_general.txt` | Electrical | 29 CFR 1926.416 — General requirements (protection of employees, electrical). <https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.416> |
| `1926.651_excavation_specific.txt` | Excavation / trenching | 29 CFR 1926.651 — Specific excavation requirements. <https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.651> |
| `1926.451_scaffolds_general.txt` | Scaffolding | 29 CFR 1926.451 — General requirements for scaffolds. <https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.451> |
| `trenching_excavation_factsheet.pdf` | Excavation / trenching | OSHA Fact Sheet, Publication 3476 — Trenching and Excavation Safety. <https://www.osha.gov/sites/default/files/publications/TRENCH_EXCAVATION_FS.pdf> |
| `struckby_hazards_instructor_guide.pdf` | Struck-by | OSHA Training Institute — Construction Focus Four: Struck-By Hazards, instructor guide. <https://www.osha.gov/sites/default/files/struckby_ig.pdf> |
| `scaffold_use_guide_osha3150.pdf` | Scaffolding | OSHA Publication 3150 — A Guide to Scaffold Use in the Construction Industry. <https://www.osha.gov/sites/default/files/publications/OSHA3150.pdf> |
| `working_safely_with_electricity_osha3942.pdf` | Electrical | OSHA Publication 3942 — Working Safely with Electricity. <https://www.osha.gov/sites/default/files/publications/OSHA3942.pdf> |

## Notes

- The five `1926.*` regulation texts were downloaded as HTML from osha.gov's
  e-CFR mirror, then stripped to plain text with BeautifulSoup (nav/header/
  footer/script tags removed, main content extracted verbatim — no
  paraphrasing or summarization of the legal text).
- The four PDFs are kept in their original PDF form; `build_rag_index.py`
  extracts their text at index time with `pypdf`.
- No document here covers electrical hazards beyond 1926.416 and the
  Publication 3942 fact sheet — Subpart K (1926.400–449) is much larger.
  1926.416 alone produced only one thin chunk and retrieved poorly against a
  natural-language electrical query during retrieval testing (ranked 4th,
  below unrelated topics); Publication 3942 was added specifically to fix
  that gap. This slice was chosen to keep the corpus small and topic-matched
  to the incident data's top risk categories (falls, electrical,
  excavation/trenching, struck-by, scaffolding) rather than to be a complete
  regulatory reference.
