"""
Anatomy-Constrained Model Wrapper for CANN-Bench.

This module implements the core "anatomy-driven AI" concept:
given a pre-trained transformer (DistilBERT) and a brain atlas,
it modulates the model's attention weights using the atlas's
neurotransmitter density gradient.

Biological motivation
---------------------
Froudist-Walsh et al. (2021) showed that a dopamine gradient across
the macaque cortex regulates access to working memory — regions with
higher dopaminergic tone sustain information longer and show stronger
task-related activity.

We operationalise this as a weight modulation scheme:
    W_constrained = W_baseline ⊙ (1 + α · (d - 0.5) · 2)

where:
    W_baseline  = pre-trained DistilBERT attention weights
    d           = dopamine density vector from atlas (∈ [0, 1])
    α           = constraint_strength (hyperparameter, default 1.0)
    ⊙           = element-wise multiplication

High-density regions amplify weights → stronger representational
access, mirroring the biological gradient.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class AnatomyConstrainedModel:
    """Wraps a HuggingFace transformer with anatomical weight modulation.

    Designed to be model-agnostic: any model with named parameters
    (accessed via .named_parameters()) can be constrained.

    Args:
        model_name: HuggingFace model identifier.
        atlas: Validated brain atlas DataFrame (from src.atlas.loader).
        constraint_strength: Scale of anatomical modulation (0=baseline, 1=full).
        num_labels: Number of classification labels.
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_name: str,
        atlas,  
        constraint_strength: float = 1.0,
        num_labels: int = 4,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.atlas = atlas
        self.constraint_strength = constraint_strength
        self.num_labels = num_labels
        self.device = device
        self._model = None

    def load(self) -> "AnatomyConstrainedModel":
        """Load the base model and apply anatomical constraints."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            raise ImportError("Install pytorch and transformers via environment.yml")

        print(f"[model] Loading {self.model_name}...")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        self._model.to(self.device)

        if self.constraint_strength > 0:
            self._apply_atlas_constraint()

        return self

    def _apply_atlas_constraint(self) -> None:
        """Modulate attention weight matrices using the brain atlas gradient.

        Targets the query, key, and value projection matrices in each
        attention layer — the weights most analogous to cortical
        connectivity patterns.
        """
        import torch
        from src.atlas.loader import compute_constraint_vector

        attention_keywords = ["q_lin", "k_lin", "v_lin", "attention"]
        modulated_layers = 0

        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if not any(kw in name for kw in attention_keywords):
                    continue
                if "weight" not in name:
                    continue

                original_shape = param.data.shape
                flat = param.data.cpu().numpy().flatten()

                constraint = compute_constraint_vector(self.atlas, len(flat))
                # Map [0,1] → gain centred at 1.0
                gain = 1.0 + self.constraint_strength * (constraint - 0.5) * 2
                modulated = flat * gain

                param.data = torch.tensor(
                    modulated.reshape(original_shape),
                    dtype=param.dtype,
                    device=self.device,
                )
                modulated_layers += 1

        print(
            f"[model] Applied dopamine gradient constraint to "
            f"{modulated_layers} attention weight matrices "
            f"(strength={self.constraint_strength})."
        )

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Call .load() before accessing .model")
        return self._model


def load_tokenizer(model_name: str):
    """Load the tokenizer matching the model."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install transformers via environment.yml")

    return AutoTokenizer.from_pretrained(model_name)
