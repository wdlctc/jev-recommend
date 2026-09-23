"""MovieLens-1M as a Jev-style Choice task.

Each record is one decision: a user's recent history (the shared *state*) and
K candidate movies, exactly one of which is the item the user actually rated
next. Splits follow the usual leave-one-out protocol: the last interaction of
every user is test, the second last is validation, earlier ones are training
targets. Negatives are sampled from movies the user never rated, with a fixed
seed per split so every model is scored on identical candidate sets.
"""
from __future__ import annotations

import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass
class MovieLens:
    titles: dict[int, str]           # movie id -> "Title (Year)"
    genres: dict[int, str]           # movie id -> "Drama, Thriller"
    sequences: dict[int, list[tuple[int, int]]]  # user -> [(movie, rating)] by time
    items: list[int]
    popularity: dict[int, int]


def download(root: str | Path = "data") -> Path:
    root = Path(root)
    target = root / "ml-1m"
    if (target / "ratings.dat").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "ml-1m.zip"
    if not archive.exists():
        urllib.request.urlretrieve(ML1M_URL, archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
    return target


def load(root: str | Path = "data") -> MovieLens:
    path = download(root)
    titles, genres = {}, {}
    for line in (path / "movies.dat").read_text(encoding="latin-1").splitlines():
        movie, title, genre = line.split("::")
        titles[int(movie)] = title
        genres[int(movie)] = genre.replace("|", ", ")
    events: dict[int, list[tuple[int, int, int]]] = {}
    popularity: dict[int, int] = {}
    for line in (path / "ratings.dat").read_text(encoding="latin-1").splitlines():
        user, movie, rating, stamp = map(int, line.split("::"))
        events.setdefault(user, []).append((stamp, movie, rating))
        popularity[movie] = popularity.get(movie, 0) + 1
    # Stable sort by timestamp; ties keep file order, which is deterministic.
    sequences = {u: [(m, r) for _, m, r in sorted(e, key=lambda x: x[0])] for u, e in events.items()}
    return MovieLens(titles, genres, sequences, sorted(popularity), popularity)


def _sample_negatives(rng: random.Random, items: list[int], seen: set[int], count: int) -> list[int]:
    out: set[int] = set()
    while len(out) < count:
        item = items[rng.randrange(len(items))]
        if item not in seen:
            out.add(item)
    return sorted(out)


def _record(data: MovieLens, user: int, target: int, history_len: int,
            num_candidates: int, rng: random.Random) -> dict:
    seq = data.sequences[user]
    history = seq[max(0, target - history_len):target]
    positive = seq[target][0]
    seen = {m for m, _ in seq}
    candidates = _sample_negatives(rng, data.items, seen, num_candidates - 1) + [positive]
    rng.shuffle(candidates)
    return {"user": user, "history": history, "candidates": candidates,
            "label": candidates.index(positive)}


def make_records(data: MovieLens, split: str, *, history_len: int = 20, num_candidates: int = 20,
                 per_user: int = 8, min_history: int = 5, seed: int = 0,
                 max_users: int | None = None) -> list[dict]:
    """Build decision records for ``split`` in {"train", "valid", "test"}.

    ``per_user`` only applies to training: that many target positions are
    drawn per user (resample by changing ``seed`` each epoch).
    """
    offset = {"train": 0, "valid": 1, "test": 2}[split]
    rng = random.Random(f"{split}-{seed}")
    users = sorted(data.sequences)
    if max_users is not None:
        users = users[:max_users]
    records = []
    for user in users:
        n = len(data.sequences[user])
        if split == "train":
            positions = list(range(min_history, n - 2))
            if not positions:
                continue
            chosen = rng.sample(positions, min(per_user, len(positions)))
        else:
            chosen = [n - 3 + offset]
            if chosen[0] < min_history:
                continue
        for target in chosen:
            records.append(_record(data, user, target, history_len, num_candidates, rng))
    return records
