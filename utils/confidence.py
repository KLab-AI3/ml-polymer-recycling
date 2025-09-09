"""Confidence calculation and visualization utilities.
Provides normalized softmax confidence and color-coded badges"""

from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F


def calculate_softmax_confidence(
    logits: torch.Tensor,
) -> Tuple[np.ndarray, float, str, str]:
    """Calculate normalized confidence using softmax
    Args:
        logits: Raw model logits tensor
    Returns:
        Tuple of (probabilities, max_confidence, confidence_level, confidence_emoji)
    """
    # ===Apply softmax to get probabilities===
    probs_np = F.softmax(logits, dim=1).cpu().numpy().flatten()

    # ===Get maximum probability as confidence===
    max_confidence = float(np.max(probs_np))

    # ===Determine confidence level and emoji===
    if max_confidence >= 0.80:
        confidence_level = "HIGH"
        confidence_emoji = "🟢"
    elif max_confidence >= 0.60:
        confidence_level = "MEDIUM"
        confidence_emoji = "🟡"
    else:
        confidence_level = "LOW"
        confidence_emoji = "🔴"

    return probs_np, max_confidence, confidence_level, confidence_emoji


def get_confidence_badge(confidence: float) -> Tuple[str, str]:
    """Get confidence badge emoji and level description
    Args:
        confidence: Confidence value (0-1)
    Returns:
        Tuple of (emoji, level)
    """
    if confidence >= 0.80:
        return "🟢", "HIGH"
    elif confidence >= 0.60:
        return "🟡", "MEDIUM"
    else:
        return "🔴", "LOW"


def format_confidence_display(confidence: float, level: str, emoji: str) -> str:
    """
    Format confidence for display in UI

    Args:
        confidence: Confidence value (0-1)
        level: Confidence level (HIGH/MEDIUM/LOW)
        emoji: Confidence emoji

    Returns:
        Formatted confidence string
    """
    return f"{emoji} **{level}** ({confidence:.1%})"


def calculate_legacy_confidence(logits_list: List[float]) -> Tuple[float, str, str]:
    """
    Calculate confidence using legacy logit margin method for backward compatibility

    Args:
        logits_list: List of raw logits

    Returns:
        Tuple of (margin, confidence_level, confidence_emoji)
    """
    if len(logits_list) < 2:
        return 0.0, "LOW", "🔴"

    logits_array = np.array(logits_list)
    sorted_logits = np.sort(logits_array)[::-1]  # Descending order
    margin = sorted_logits[0] - sorted_logits[1]

    # ===Define thresholds for margin-based confidence===
    if margin >= 2.0:
        confidence_level = "HIGH"
        confidence_emoji = "🟢"
    elif margin >= 1.0:
        confidence_level = "MEDIUM"
        confidence_emoji = "🟡"
    else:
        confidence_level = "LOW"
        confidence_emoji = "🔴"

    return margin, confidence_level, confidence_emoji
