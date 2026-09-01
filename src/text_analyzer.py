from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Coefficient inspection (see docs/model_card.md) found calendar-year and month
# tokens dominating the top TF-IDF coefficients -- a scrape-selection/dataset-
# construction confound (which years' narratives got scraped correlates with
# label), not real risk signal. Excluded as stop words so bigrams built on top
# of them (e.g. "2007 employee") are also suppressed at the tokenization stage.
MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
YEAR_TOKENS = {str(y) for y in range(1900, 2031)}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS | MONTH_NAMES | YEAR_TOKENS)


class ArtifactFilteringAnalyzer:
    """TF-IDF analyzer that also drops artifact bigrams: two identical words
    (e.g. "employee employee", produced when stopword removal collapses
    "Employee #1 and Employee #2" into adjacent duplicate tokens) and bigrams
    where one token is a bare number under 100 (e.g. "20 employee", from
    "Employee #20") -- both are ID-numbering artifacts, not risk signal.

    Lives in its own module (rather than as a closure, or inline in whichever
    script trains the model) so a fitted vectorizer that stores this as its
    `analyzer` can be pickled/joblib-dumped from one script and unpickled from
    another -- pickle records a class by its module path, and a class defined
    in a script run directly is recorded under the unimportable name
    `__main__`."""

    def __init__(self, stop_words=CUSTOM_STOP_WORDS):
        self._base_analyzer = TfidfVectorizer(
            stop_words=stop_words, ngram_range=(1, 2)
        ).build_analyzer()

    def __call__(self, doc):
        tokens = []
        for token in self._base_analyzer(doc):
            words = token.split(" ")
            if len(words) == 2:
                a, b = words
                if a == b:
                    continue
                if any(w.isdigit() and int(w) < 100 for w in words):
                    continue
            tokens.append(token)
        return tokens


def build_custom_analyzer():
    return ArtifactFilteringAnalyzer(CUSTOM_STOP_WORDS)
