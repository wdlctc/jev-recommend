"""Render records into token segments: one shared state, K candidate segments.

The same token ids are used by every scoring mode, so the pointwise and the
isolated-mask layouts see bit-identical inputs per candidate. Candidate
segments start with a newline so tokenizing them separately cannot merge
tokens across the state/candidate boundary.
"""
from __future__ import annotations

from .data import MovieLens

STATE_HEADER = "A user rated these movies, oldest to newest (1-5 stars):\n"
QUESTION = "Question: will the user watch the following movie next?\n"
CANDIDATE = "\nCandidate: {title} [{genres}]\nAnswer Yes or No:"
DECIDE = "\nAmong all candidates above, the movie the user watches next is"


def state_text(data: MovieLens, history: list[tuple[int, int]]) -> str:
    lines = [f"- {data.titles[m]} [{data.genres[m]}]: {r:g}" for m, r in history]
    return STATE_HEADER + "\n".join(lines) + "\n" + QUESTION


def candidate_text(data: MovieLens, movie: int) -> str:
    return CANDIDATE.format(title=data.titles[movie], genres=data.genres[movie])


class Encoder:
    """Tokenize records once; models consume the resulting id segments."""

    def __init__(self, tokenizer, data: MovieLens):
        self.tokenizer, self.data = tokenizer, data
        self._cand_cache: dict[int, list[int]] = {}
        self.decide_ids = self._ids(DECIDE)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __call__(self, record: dict) -> dict:
        cands = []
        for movie in record["candidates"]:
            if movie not in self._cand_cache:
                self._cand_cache[movie] = self._ids(candidate_text(self.data, movie))
            cands.append(self._cand_cache[movie])
        return {"state": self._ids(state_text(self.data, record["history"])),
                "cands": cands, "decide": self.decide_ids, "label": record["label"]}
