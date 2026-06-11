from __future__ import annotations

import re

from ...models import VideoMetadata

# Words that look like proper nouns only because they start a sentence or a title.
_STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how", "in",
    "into", "is", "it", "its", "of", "on", "or", "our", "that", "the", "their", "this",
    "to", "was", "we", "what", "when", "where", "which", "why", "with", "you", "your",
    # Russian
    "и", "в", "во", "не", "на", "но", "как", "что", "это", "этот", "эта", "для", "по",
    "из", "за", "от", "до", "или", "так", "же", "вы", "мы", "он", "она", "они", "его",
    "ее", "их", "при", "про", "у", "о", "об", "к", "с", "со",
}

# CamelCase, dotted, versioned, or otherwise technical-looking tokens.
_TECHNICAL_TOKEN = re.compile(
    r"^(?:[A-Za-z][a-z0-9]*[A-Z][A-Za-z0-9]*|[A-Za-z][\w-]*\.[\w.-]+|[A-Za-z][\w-]*\d[\w.-]*|v\d[\w.]*)$"
)
_CAPITALIZED_TOKEN = re.compile(r"^[A-ZА-ЯЁ][\w'-]+$", re.UNICODE)
_WORD = re.compile(r"[\w.'-]+", re.UNICODE)


def extract_keyterms(metadata: VideoMetadata, *, max_terms: int = 200) -> list[str]:
    """Collect likely proper nouns and technical terms for STT keyterm boosting.

    Sources are scanned in priority order (title, chapters, tags, description) so
    truncation at `max_terms` keeps the most trustworthy terms.
    """
    if max_terms <= 0:
        return []
    sources = [
        metadata.title or "",
        " . ".join(chapter.title for chapter in metadata.chapters),
        " . ".join(metadata.tags),
        metadata.description or "",
    ]
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip().strip(".,;:!?\"'()[]{}«»")
        if len(cleaned) < 3 or len(cleaned) > 50:
            return
        key = cleaned.casefold()
        if key in seen or key in _STOPWORDS:
            return
        seen.add(key)
        terms.append(cleaned)

    for source in sources:
        for phrase in _proper_phrases(source):
            add(phrase)
        for token in _WORD.findall(source):
            if _TECHNICAL_TOKEN.match(token):
                add(token)
            elif _CAPITALIZED_TOKEN.match(token) and token.casefold() not in _STOPWORDS:
                add(token)
        if len(terms) >= max_terms:
            break
    return terms[:max_terms]


def _proper_phrases(text: str) -> list[str]:
    """Find runs of 2-3 adjacent capitalized words ("Visual Studio Code")."""
    phrases: list[str] = []
    tokens = _WORD.findall(text)
    run: list[str] = []
    for token in tokens:
        if _CAPITALIZED_TOKEN.match(token) and token.casefold() not in _STOPWORDS:
            run.append(token)
            continue
        if len(run) >= 2:
            phrases.append(" ".join(run[:3]))
        run = []
    if len(run) >= 2:
        phrases.append(" ".join(run[:3]))
    return phrases
