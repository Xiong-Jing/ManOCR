from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


def bounded_edit_distance(a: str, b: str, max_distance: int) -> int:
    """
    Levenshtein distance with early cutoff.
    Returns max_distance + 1 when the distance is definitely larger.
    """
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    if a == b:
        return 0

    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        curr = [i]
        row_min = curr[0]

        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (ca != cb)
            value = min(insert_cost, delete_cost, replace_cost)
            curr.append(value)
            row_min = min(row_min, value)

        if row_min > max_distance:
            return max_distance + 1

        prev = curr

    return prev[-1]


def load_lexicon_from_manifest(manifest_path: str | Path) -> Counter:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    counter: Counter = Counter()

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid manifest line {line_idx} in {manifest_path}: {repr(line)}"
                )

            label = parts[1].strip()

            if label:
                counter[label] += 1

    return counter


class LexiconCorrector:
    """
    Correct decoded words to the nearest training-set lexicon entry.

    This is intended as OCR post-processing. Use only train-set labels as the
    lexicon to avoid validation/test leakage.
    """

    def __init__(
        self,
        lexicon_counts: Counter | Dict[str, int],
        max_edit_distance: int = 2,
        length_delta: int = 2,
        min_word_length: int = 2,
    ):
        self.lexicon_counts = Counter(lexicon_counts)
        self.max_edit_distance = max_edit_distance
        self.length_delta = length_delta
        self.min_word_length = min_word_length
        self._cache: Dict[str, str] = {}

        self.by_length: Dict[int, List[str]] = defaultdict(list)

        for word in self.lexicon_counts:
            self.by_length[len(word)].append(word)

        for length in self.by_length:
            self.by_length[length].sort(key=lambda item: (-self.lexicon_counts[item], item))

    def correct(self, word: str) -> str:
        if word in self._cache:
            return self._cache[word]

        if len(word) < self.min_word_length:
            self._cache[word] = word
            return word

        if word in self.lexicon_counts:
            self._cache[word] = word
            return word

        best_word = word
        best_distance = self.max_edit_distance + 1
        best_frequency = -1

        min_len = max(1, len(word) - self.length_delta)
        max_len = len(word) + self.length_delta

        for length in range(min_len, max_len + 1):
            for candidate in self.by_length.get(length, []):
                distance = bounded_edit_distance(
                    word,
                    candidate,
                    max_distance=self.max_edit_distance,
                )

                if distance > self.max_edit_distance:
                    continue

                frequency = self.lexicon_counts[candidate]

                if (
                    distance < best_distance
                    or (distance == best_distance and frequency > best_frequency)
                    or (
                        distance == best_distance
                        and frequency == best_frequency
                        and candidate < best_word
                    )
                ):
                    best_word = candidate
                    best_distance = distance
                    best_frequency = frequency

        self._cache[word] = best_word
        return best_word

    def correct_batch(self, words: Iterable[str]) -> List[str]:
        return [self.correct(word) for word in words]


class CTCLexiconReranker:
    """
    Rerank nearby lexicon candidates with CTC sequence likelihood.

    Unlike nearest-word correction, this can change a valid greedy prediction
    into another valid lexicon word when the frame probabilities support it.
    """

    def __init__(
        self,
        lexicon_counts: Counter | Dict[str, int],
        char_to_idx: Dict[str, int],
        blank_idx: int = 0,
        max_edit_distance: int = 2,
        length_delta: int = 2,
        min_word_length: int = 2,
        max_candidates: int = 40,
        prior_weight: float = 0.05,
    ):
        self.lexicon_counts = Counter(lexicon_counts)
        self.char_to_idx = char_to_idx
        self.blank_idx = blank_idx
        self.max_edit_distance = max_edit_distance
        self.length_delta = length_delta
        self.min_word_length = min_word_length
        self.max_candidates = max_candidates
        self.prior_weight = prior_weight

        self.by_length: Dict[int, List[str]] = defaultdict(list)
        for word in self.lexicon_counts:
            if all(char in self.char_to_idx for char in word):
                self.by_length[len(word)].append(word)

        for length in self.by_length:
            self.by_length[length].sort(key=lambda item: (-self.lexicon_counts[item], item))

        self._candidate_cache: Dict[str, List[str]] = {}
        self._target_cache: Dict[str, torch.Tensor] = {}

    def get_candidates(self, word: str) -> List[str]:
        if word in self._candidate_cache:
            return self._candidate_cache[word]

        if len(word) < self.min_word_length:
            self._candidate_cache[word] = [word]
            return self._candidate_cache[word]

        scored = []
        min_len = max(1, len(word) - self.length_delta)
        max_len = len(word) + self.length_delta

        for length in range(min_len, max_len + 1):
            for candidate in self.by_length.get(length, []):
                distance = bounded_edit_distance(
                    word,
                    candidate,
                    max_distance=self.max_edit_distance,
                )
                if distance <= self.max_edit_distance:
                    scored.append((distance, -self.lexicon_counts[candidate], candidate))

        if not scored and word:
            scored.append((0, 0, word))

        scored.sort()
        candidates = [candidate for _, _, candidate in scored[: self.max_candidates]]

        if word in self.lexicon_counts and word not in candidates:
            candidates.insert(0, word)

        self._candidate_cache[word] = candidates
        return candidates

    def word_to_target(self, word: str, device: torch.device) -> torch.Tensor | None:
        if word in self._target_cache:
            return self._target_cache[word].to(device)

        indices = []
        for char in word:
            idx = self.char_to_idx.get(char)
            if idx is None:
                return None
            indices.append(idx)

        target = torch.tensor(indices, dtype=torch.long)
        self._target_cache[word] = target
        return target.to(device)

    def score_word(self, log_probs: torch.Tensor, word: str) -> torch.Tensor:
        target = self.word_to_target(word, log_probs.device)
        if target is None or target.numel() == 0:
            return log_probs.new_tensor(float("-inf"))

        ext_symbols = [self.blank_idx]
        for idx in target.tolist():
            ext_symbols.append(int(idx))
            ext_symbols.append(self.blank_idx)

        symbols = torch.tensor(ext_symbols, dtype=torch.long, device=log_probs.device)
        num_states = symbols.numel()

        alpha = log_probs.new_full((num_states,), float("-inf"))
        alpha[0] = log_probs[0, self.blank_idx]
        if num_states > 1:
            alpha[1] = log_probs[0, symbols[1]]

        for t in range(1, log_probs.shape[0]):
            prev_alpha = alpha
            new_alpha = log_probs.new_full((num_states,), float("-inf"))

            for s in range(num_states):
                sources = [prev_alpha[s]]
                if s - 1 >= 0:
                    sources.append(prev_alpha[s - 1])
                if (
                    s - 2 >= 0
                    and symbols[s] != self.blank_idx
                    and symbols[s] != symbols[s - 2]
                ):
                    sources.append(prev_alpha[s - 2])

                source_score = torch.logsumexp(torch.stack(sources), dim=0)
                new_alpha[s] = source_score + log_probs[t, symbols[s]]

            alpha = new_alpha

        if num_states == 1:
            score = alpha[0]
        else:
            score = torch.logsumexp(alpha[-2:], dim=0)

        if self.prior_weight > 0:
            frequency = float(self.lexicon_counts.get(word, 1))
            score = score + self.prior_weight * torch.log(
                log_probs.new_tensor(frequency)
            )

        return score

    def rerank_one(self, log_probs: torch.Tensor, raw_word: str) -> str:
        candidates = self.get_candidates(raw_word)

        if not candidates:
            return raw_word

        best_word = raw_word
        best_score = log_probs.new_tensor(float("-inf"))

        for candidate in candidates:
            score = self.score_word(log_probs, candidate)

            if score > best_score:
                best_score = score
                best_word = candidate

        return best_word

    def rerank_batch(self, logits: torch.Tensor, raw_words: List[str]) -> List[str]:
        if logits.dim() != 3:
            raise ValueError(f"logits should be [B, T, C], got {tuple(logits.shape)}")

        log_probs = F.log_softmax(logits.float(), dim=-1).detach().cpu()
        return [
            self.rerank_one(item_log_probs, raw_word)
            for item_log_probs, raw_word in zip(log_probs, raw_words)
        ]
