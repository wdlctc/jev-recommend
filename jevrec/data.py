"""MovieLens-1M as a Jev-style Choice task.

Each record is one decision: a user's recent history (the shared *state*) and
K candidate movies, exactly one of which is the item the user actually rated
next. Splits follow the usual leave-one-out protocol: the last interaction of
every user is test, the second last is validation, earlier ones are training
targets. Negatives are sampled from movies the user never rated, with a fixed
seed per split so every model is scored on identical candidate sets.
"""
from __future__ import annotations

import bisect
import csv
import itertools
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    # 2018 snapshot: ~9.7k movies, most released after ML-1M's 2000 cut-off.
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}


@dataclass
class MovieLens:
    titles: dict[int, str]           # movie id -> "Title (Year)"
    genres: dict[int, str]           # movie id -> "Drama, Thriller"
    sequences: dict[int, list[tuple[int, float]]]  # user -> [(movie, rating)] by time
    items: list[int]
    popularity: dict[int, int]


def download(root: str | Path = "data", name: str = "ml-1m") -> Path:
    root = Path(root)
    target = root / name
    if target.exists() and any(target.glob("ratings.*")):
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{name}.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URLS[name], archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
    return target


def _rows(path: Path):
    """Yield split rows from ML-1M '::' files or ml-latest '.csv' files."""
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            yield from reader
    else:
        for line in path.read_text(encoding="latin-1").splitlines():
            yield line.split("::")


def load(root: str | Path = "data", name: str = "ml-1m") -> MovieLens:
    path = download(root, name)
    ext = ".csv" if name != "ml-1m" else ".dat"
    titles, genres = {}, {}
    for movie, title, genre in _rows(path / f"movies{ext}"):
        titles[int(movie)] = title.strip()
        genres[int(movie)] = genre.replace("|", ", ")
    events: dict[int, list[tuple[int, int, float]]] = {}
    popularity: dict[int, int] = {}
    for user, movie, rating, stamp in _rows(path / f"ratings{ext}"):
        user, movie, rating, stamp = int(user), int(movie), float(rating), int(stamp)
        events.setdefault(user, []).append((stamp, movie, rating))
        popularity[movie] = popularity.get(movie, 0) + 1
    # Stable sort by timestamp; ties keep file order, which is deterministic.
    sequences = {u: [(m, r) for _, m, r in sorted(e, key=lambda x: x[0])] for u, e in events.items()}
    return MovieLens(titles, genres, sequences, sorted(popularity), popularity)


def _sample_negatives(rng: random.Random, items: list[int], seen: set[int], count: int,
                      cum_weights: list[float] | None = None) -> list[int]:
    """Uniform over unseen items, or popularity-weighted when ``cum_weights`` is given."""
    out: set[int] = set()
    while len(out) < count:
        if cum_weights is None:
            item = items[rng.randrange(len(items))]
        else:
            item = items[bisect.bisect_right(cum_weights, rng.random() * cum_weights[-1])]
        if item not in seen:
            out.add(item)
    return sorted(out)


def _record(data: MovieLens, user: int, target: int, history_len: int,
            num_candidates: int, rng: random.Random, cum_weights=None) -> dict:
    seq = data.sequences[user]
    history = seq[max(0, target - history_len):target]
    positive = seq[target][0]
    seen = {m for m, _ in seq}
    candidates = _sample_negatives(rng, data.items, seen, num_candidates - 1, cum_weights) + [positive]
    rng.shuffle(candidates)
    return {"user": user, "history": history, "candidates": candidates,
            "label": candidates.index(positive)}


def make_records(data: MovieLens, split: str, *, history_len: int = 20, num_candidates: int = 20,
                 per_user: int = 8, min_history: int = 5, seed: int = 0,
                 max_users: int | None = None, negatives: str = "uniform") -> list[dict]:
    """Build decision records for ``split`` in {"train", "valid", "test"}.

    ``per_user`` only applies to training: that many target positions are
    drawn per user (resample by changing ``seed`` each epoch). ``negatives`` is
    "uniform" or "popular" (sampled proportionally to interaction count, i.e.
    harder, more plausible distractors).
    """
    offset = {"train": 0, "valid": 1, "test": 2}[split]
    rng = random.Random(f"{split}-{seed}" if negatives == "uniform" else f"{split}-{seed}-{negatives}")
    cum_weights = None
    if negatives == "popular":
        cum_weights = list(itertools.accumulate(data.popularity[m] for m in data.items))
    elif negatives != "uniform":
        raise ValueError("negatives must be 'uniform' or 'popular'")
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
            records.append(_record(data, user, target, history_len, num_candidates, rng, cum_weights))
    return records
