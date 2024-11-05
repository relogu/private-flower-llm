"""Metrics to use with MosaicML Composer."""

from llmfoundry.registry import metrics
from flower_llm.metrics.unigram_normalized_metrics import (
    UnigramNormalizedLanguageCrossEntropy,
    UnigramNormalizedLanguagePerplexity,
)

metrics.register(
    "unigram_normalized_language_cross_entropy",
    func=UnigramNormalizedLanguageCrossEntropy,
)

metrics.register(
    "unigram_normalized_language_perplexity",
    func=UnigramNormalizedLanguagePerplexity,
)
