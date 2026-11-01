from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
DEFAULT_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"


class NLIScorer(ABC):
    """
    Minimal abstract interface for NLI scoring.
    Allows easy mocking in fast unit tests without loading neural weights.
    """

    @abstractmethod
    def score(self, pairs: List[Tuple[str, str]]) -> List[Tuple[float, float, float]]:
        """
        Score a list of (premise, hypothesis) string pairs.
        Returns a list of (p_entailment, p_neutral, p_contradiction) tuples,
        where each triple represents softmax probabilities summing to 1.0.
        """
        pass


class DebertaNLIScorer(NLIScorer):
    """
    Concrete implementation of NLIScorer using cross-encoder/nli-deberta-v3-small.
    Reads label ordering dynamically from model.config.id2label.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str = DEFAULT_REVISION,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading NLI model '{model_name}' (rev {revision}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision
        ).to(self.device)
        self.model.eval()

        # Dynamically read id2label mapping instead of hardcoding label indices
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        try:
            self.ent_idx = next(k for k, v in id2label.items() if "entail" in v)
            self.neu_idx = next(k for k, v in id2label.items() if "neut" in v)
            self.con_idx = next(k for k, v in id2label.items() if "contra" in v)
        except StopIteration as e:
            raise ValueError(f"Could not resolve entail/neutral/contradict from config id2label: {id2label}") from e

        logger.info(f"NLI label indices: entail={self.ent_idx}, neutral={self.neu_idx}, contra={self.con_idx}")

    def score(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[Tuple[float, float, float]]:
        if not pairs:
            return []

        results: List[Tuple[float, float, float]] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            for b_idx in range(len(batch)):
                p_ent = float(probs[b_idx, self.ent_idx].item())
                p_neu = float(probs[b_idx, self.neu_idx].item())
                p_con = float(probs[b_idx, self.con_idx].item())
                results.append((p_ent, p_neu, p_con))

        return results

