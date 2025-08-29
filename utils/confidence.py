"""Confidence calculation and visualization utilities.
Provides normalized softmax confidence and color-coded badges"""
from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F


def calculate_softmax_confidence(logits: torch.Tensor) -> Tuple[np.ndarray, float, str, str]:
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


def create_confidence_progress_html(
    probabilities: np.ndarray,
    labels: List[str],
    highlight_idx: int
) -> str:
    """
    Create HTML for confidence progress bars

    Args:
        probabilities: Array of class probabilities
        labels: List of class labels
        highlight_idx: Index of predicted class to highlight

    Returns:
        HTML string for progress bars
    """
    if len(probabilities) == 0 or len(labels) == 0:
        return "<p>No confidence data available</p>"

    html_parts = []

    for i, (prob, label) in enumerate(zip(probabilities, labels)):
        # ===Color based on whether this is the predicted class===
        if i == highlight_idx:
            if prob >= 0.80:
                color = "#22c55e"  # green-500
                text_color = "#ffffff"
            elif prob >= 0.60:
                color = "#eab308"  # yellow-500
                text_color = "#000000"
            else:
                color = "#ef4444"  # red-500
                text_color = "#ffffff"
        else:
            color = "#e5e7eb"  # gray-200
            text_color = "#6b7280"  # gray-500

        percentage = prob * 100

        html_parts.append(f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 0.875rem; font-weight: 500; color: #374151;">{label}</span>
                <span style="font-size: 0.875rem; color: #6b7280;">{percentage:.1f}%</span>
            </div>
            <div style="width: 100%; background-color: #f3f4f6; border-radius: 0.375rem; height: 20px; overflow: hidden;">
                <div style="
                    width: {percentage}%; 
                    height: 100%; 
                    background-color: {color}; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    transition: width 0.3s ease;
                ">
                    {f'<span style="color: {text_color}; font-size: 0.75rem; font-weight: 600;">{percentage:.1f}%</span>' if percentage > 20 else ''}
                </div>
            </div>
        </div>
        """)

    return f""" 
    <div style="padding: 16px; background-color: #f9fafb; border-radius: 0.5rem; border: 1px solid #e5e7eb;">
        <h4 style="margin: 0 0 12px 0; font-size: 1rem; color: #374151;">Confidence Breakdown</h4>
        {''.join(html_parts)}
    </div>
    """


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
