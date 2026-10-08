from __future__ import annotations
import random
from .text import normalize

# Hand-written source material, not scraped books. Every generated example has
# words spelled out and a known target. Expand with your own licensed text files.
NOUNS = ["cat", "dog", "bird", "car", "boat", "house", "garden", "window", "river", "road", "tree", "door"]
ADJECTIVES = ["red", "blue", "green", "small", "large", "old", "new", "bright", "quiet", "warm"]
VERBS = ["see", "find", "move", "watch", "paint", "follow"]
COMMANDS = ["turn on the light", "turn off the light", "open the door", "close the door", "stop the music", "play the music",
            "save the file", "read the next page", "go back", "start again", "speak more slowly", "show the time"]


def generated_texts(count: int, seed: int = 17) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    result, seen = [], set()
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 100 + 10000:
            raise ValueError("Generator vocabulary exhausted; add books or larger text sources")
        n, a, v = rng.choice(NOUNS), rng.choice(ADJECTIVES), rng.choice(VERBS)
        pattern = rng.randrange(5)
        if pattern == 0:
            text = f"the {a} {n}"
        elif pattern == 1:
            text = f"{v} the {a} {n}"
        elif pattern == 2:
            text = f"we can {v} a {n}"
        elif pattern == 3:
            text = f"the {n} is {a}"
        else:
            text = rng.choice(COMMANDS)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def sampling_weights(rows: list[dict], difficulty: dict[str, float]) -> list[float]:
    # At least 30% uniform sampling. Difficult utterances cannot starve the rest.
    values = [1 + min(3, max(0, difficulty.get(r["id"], 0))) for r in rows]
    total = sum(values)
    return [0.3 / len(rows) + 0.7 * v / total for v in values]


def update_difficulty(difficulty: dict[str, float], row_id: str, error: float) -> None:
    old = difficulty.get(row_id, error)
    difficulty[row_id] = float(0.8 * old + 0.2 * min(3, max(0, error)))
