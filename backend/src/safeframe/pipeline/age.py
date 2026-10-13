"""Apparent-age estimation from face crops.

The default model is a ViT classifier fine-tuned on FairFace, which predicts a
distribution over age ranges (0-2, 3-9, 10-19, 20-29, ...). We convert that
distribution into P(under 18) by assuming ages are uniform within each range,
so the "10-19" bucket contributes 8/10 of its probability mass.

This estimates *apparent* age from appearance. It is probabilistic, has known
demographic error patterns, and must never be treated as a fact about a person.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from safeframe.pipeline.policy import bucket_stats

DEFAULT_AGE_MODEL = "nateraw/vit-age-classifier"


class ViTAgeEstimator:
    def __init__(
        self,
        model_id: str = DEFAULT_AGE_MODEL,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, revision=revision)
        self.model.to(self.device).eval()
        labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        fractions, mids = bucket_stats(labels)
        self._fractions = torch.tensor(fractions, device=self.device)
        self._mids = torch.tensor(mids, device=self.device)
        self.labels = labels

    def estimate(self, faces: Sequence[np.ndarray]) -> list[tuple[float, float]]:
        if not faces:
            return []
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in faces]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            probs = self.model(**inputs).logits.softmax(dim=-1)
        p_minor = (probs * self._fractions).sum(dim=-1).tolist()
        expected = (probs * self._mids).sum(dim=-1).tolist()
        return list(zip(p_minor, expected, strict=True))
