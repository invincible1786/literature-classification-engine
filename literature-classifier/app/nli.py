"""NLI Scoring module providing abstract interface, DeBERTa CrossEncoder, and StubScorer."""
import hashlib
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
DEFAULT_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"


class NLIScorer(ABC):
    """
    Abstract interface for Natural Language Inference scoring.
    Enables pluggable model architectures and deterministic unit testing via StubScorer.
    """

    @abstractmethod
    def score(self, pairs: List[Tuple[str, str]]) -> List[Tuple[float, float, float]]:
        """
        Score a list of (premise, hypothesis) string pairs.
        Returns a list of (p_entailment, p_neutral, p_contradiction) triples summing to 1.0.
        """
        pass

    def score_margins(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Compute the directional NLI margin m = p_entailment - p_contradiction in [-1.0, +1.0].
        """
        triples = self.score(pairs)
        return [p_ent - p_con for p_ent, _, p_con in triples]

    def predict_proba(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Convenience wrapper returning dictionary probabilities for entailment, neutral, contradiction.
        """
        triples = self.score(pairs)
        return [
            {"entailment": t[0], "neutral": t[1], "contradiction": t[2]}
            for t in triples
        ]


class StubScorer(NLIScorer):
    """
    Deterministic in-memory StubScorer for Tier 1 unit testing.
    Executes without downloading or running neural transformer models.
    """

    def __init__(
        self,
        canned_scores: Optional[Dict[Tuple[str, str], Tuple[float, float, float]]] = None,
        default_margin: float = 0.0
    ):
        self.canned_scores = canned_scores or {}
        self.default_margin = default_margin
        self.device = "cpu"
        self.model_name = "stub-scorer"
        self.revision = "stub-rev-1.0"
        self.call_count = 0
        self.pairs_scored = 0

    def score(self, pairs: List[Tuple[str, str]]) -> List[Tuple[float, float, float]]:
        self.call_count += 1
        self.pairs_scored += len(pairs)
        results: List[Tuple[float, float, float]] = []

        for premise, hypothesis in pairs:
            # Check canned exact match
            if (premise, hypothesis) in self.canned_scores:
                results.append(self.canned_scores[(premise, hypothesis)])
                continue

            # Deterministic heuristic fallback based on textual content
            p_lower = premise.lower()
            h_lower = hypothesis.lower()

            # Contrast detection
            if any(reagent_word in p_lower for reagent_word in ["starting material", "reagent", "standard", "purchased from"]):
                if any(contra_word in h_lower for contra_word in ["starting material", "precursor", "reagent", "standard"]):
                    results.append((0.98, 0.01, 0.01))
                    continue
                elif any(synth_word in h_lower for synth_word in ["synthes", "produc", "extract"]):
                    results.append((0.05, 0.90, 0.05))
                    continue

            # Positive production detection
            if any(prod_word in p_lower for prod_word in ["produced", "synthesized", "extracted", "isolated", "yield"]):
                if any(synth_word in h_lower for synth_word in ["synthes", "produc", "extract", "isolat"]):
                    results.append((0.95, 0.04, 0.01))
                    continue

            # Fallback based on default_margin
            p_ent = max(0.0, min(1.0, (self.default_margin + 1.0) / 2.0))
            p_con = max(0.0, min(1.0, 1.0 - p_ent))
            results.append((p_ent, 0.0, p_con))

        return results


class DebertaNLIScorer(NLIScorer):
    """
    Production Cross-Encoder implementation using cross-encoder/nli-deberta-v3-small.
    Features:
    - Pinned model commit hash (revision).
    - Dynamic label configuration introspection via model.config.id2label.
    - LRU margin cache to eliminate redundant forward passes on identical pairs.
    - Dynamic batched inference with torch.inference_mode().
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str = DEFAULT_REVISION,
        device: Optional[str] = None,
        max_cache_size: int = 4096
    ):
        self.model_name = model_name
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, Tuple[float, float, float]] = {}
        self.inference_pairs = 0
        self.cache_hits = 0

        logger.info(f"Loading NLI model '{model_name}' (rev {revision[:8]}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision
        ).to(self.device)
        self.model.eval()

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        try:
            self.ent_idx = next(k for k, v in id2label.items() if "entail" in v)
            self.neu_idx = next(k for k, v in id2label.items() if "neut" in v)
            self.con_idx = next(k for k, v in id2label.items() if "contra" in v)
        except StopIteration as e:
            raise ValueError(f"Could not resolve entail/neutral/contradict from config id2label: {id2label}") from e

        logger.info(f"NLI label indices: entail={self.ent_idx}, neutral={self.neu_idx}, contra={self.con_idx}")

    def _cache_key(self, premise: str, hypothesis: str) -> str:
        h1 = hashlib.sha1(premise.encode("utf-8")).hexdigest()[:16]
        h2 = hashlib.sha1(hypothesis.encode("utf-8")).hexdigest()[:16]
        return f"{h1}:{h2}:{self.revision[:8]}"

    def score(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[Tuple[float, float, float]]:
        if not pairs:
            return []

        results: List[Optional[Tuple[float, float, float]]] = [None] * len(pairs)
        uncached_indices: List[int] = []
        uncached_pairs: List[Tuple[str, str]] = []

        # Check LRU cache
        for idx, (p, h) in enumerate(pairs):
            k = self._cache_key(p, h)
            if k in self._cache:
                results[idx] = self._cache[k]
                self.cache_hits += 1
            else:
                uncached_indices.append(idx)
                uncached_pairs.append((p, h))

        if uncached_pairs:
            self.inference_pairs += len(uncached_pairs)
            for i in range(0, len(uncached_pairs), batch_size):
                batch = uncached_pairs[i : i + batch_size]
                batch_indices = uncached_indices[i : i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)

                with torch.inference_mode():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)

                for b_sub_idx in range(len(batch)):
                    p_ent = float(probs[b_sub_idx, self.ent_idx].item())
                    p_neu = float(probs[b_sub_idx, self.neu_idx].item())
                    p_con = float(probs[b_sub_idx, self.con_idx].item())
                    res = (p_ent, p_neu, p_con)

                    orig_idx = batch_indices[b_sub_idx]
                    results[orig_idx] = res

                    # Cache result
                    k = self._cache_key(batch[b_sub_idx][0], batch[b_sub_idx][1])
                    if len(self._cache) < self.max_cache_size:
                        self._cache[k] = res

        return [r for r in results if r is not None]
