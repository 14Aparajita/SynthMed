"""
Grammar-constrained JSON generation for SynthMed.
Uses lm-format-enforcer to force the LLM to emit schema-valid JSON.

If lm-format-enforcer is not installed, falls back to unconstrained decoding
(this preserves reproducibility on environments without the package).
"""

import json
import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("synthmed.generation")

try:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )
    HAS_LMFE = True
except ImportError:
    HAS_LMFE = False
    logger.warning(
        "lm-format-enforcer not installed. "
        "Constrained decoding disabled; falling back to standard generation. "
        "Install with: pip install lm-format-enforcer"
    )


class ConstrainedMetadataGenerator:
    """
    Wraps a HuggingFace causal LM to produce schema-valid JSON via
    grammar-constrained decoding.
    """

    def __init__(
        self,
        model,
        tokenizer,
        schema: Dict,
        device: str = "cpu",
        max_length: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.schema = schema
        self.device = device
        self.max_length = max_length
        self.available = HAS_LMFE

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
    ) -> List[str]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length // 2,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_length,
            "do_sample": temperature > 0.0,
            "temperature": max(temperature, 1e-5),
            "top_p": 0.95,
            "top_k": 50,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.available and num_return_sequences == 1:
            try:
                parser = JsonSchemaParser(self.schema)
                prefix_fn = build_transformers_prefix_allowed_tokens_fn(
                    self.tokenizer, parser
                )
                gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
            except Exception as e:
                logger.warning(f"Constrained decoding setup failed: {e}. Falling back.")

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]