"""PDF zone marker extraction utility.

This module implements the "通用 PDF 图纸图区标识提取算法" described in the
user specification.  It analyses a single-page engineering drawing PDF and
detects the four groups of border markers (top, bottom, left, right).  The
implementation focuses on robustness and configurability so it can operate on
pages of different sizes, rotations, and typographic styles.

Typical usage::

    from zone_marker_extractor import extract_zone_markers

    result = extract_zone_markers("drawing.pdf")
    for direction, payload in result.items():
        print(direction, [item["text"] for item in payload["labels"]])

The returned ``result`` dictionary contains a ``labels`` list that stores the
detected marker information (content, position, and font size) and a ``coverage``
value which indicates the fraction of candidate labels that belong to the
recovered continuous sequence.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTChar, LTTextContainer, LTTextLine


@dataclass
class TextLabel:
    """Representation of a filtered text fragment on the page."""

    text: str
    bbox: Tuple[float, float, float, float]
    font_size: float

    @property
    def position(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


@dataclass
class DirectionResult:
    labels: List[TextLabel]
    coverage: float

    def to_payload(self) -> Dict[str, object]:
        return {
            "labels": [
                {
                    "text": label.text,
                    "position": label.position,
                    "font_size": label.font_size,
                }
                for label in self.labels
            ],
            "coverage": self.coverage,
        }


AllowedChar = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def extract_zone_markers(
    pdf_path: str | Path,
    *,
    margin_ratio: float = 0.08,
    min_font: float = 2.0,
    band_ratio: float = 0.05,
    debug: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Extract the four border marker sequences from a PDF drawing.

    Args:
        pdf_path: Path to a PDF file containing a single drawing page.
        margin_ratio: Fraction of the page dimension used when gathering the
            initial set of candidate labels near a particular edge.
        min_font: Font size threshold for filtering out very small text noise.
        band_ratio: Controls the adaptive width of the density band used to
            locate the border marker strip.
        debug: When ``True`` the function will save a ``debug_markers.png``
            visualisation in the current working directory.

    Returns:
        A dictionary with keys ``"top"``, ``"bottom"``, ``"left"``, and
        ``"right"``.  Each entry is itself a dictionary containing the detected
        ``labels`` list and the ``coverage`` ratio.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    labels, page_width, page_height = _extract_labels(pdf_path, min_font)

    results = {
        "top": _recover_direction(
            labels,
            axis="horizontal",
            edge="top",
            page_width=page_width,
            page_height=page_height,
            margin_ratio=margin_ratio,
            band_ratio=band_ratio,
        ),
        "bottom": _recover_direction(
            labels,
            axis="horizontal",
            edge="bottom",
            page_width=page_width,
            page_height=page_height,
            margin_ratio=margin_ratio,
            band_ratio=band_ratio,
        ),
        "left": _recover_direction(
            labels,
            axis="vertical",
            edge="left",
            page_width=page_width,
            page_height=page_height,
            margin_ratio=margin_ratio,
            band_ratio=band_ratio,
        ),
        "right": _recover_direction(
            labels,
            axis="vertical",
            edge="right",
            page_width=page_width,
            page_height=page_height,
            margin_ratio=margin_ratio,
            band_ratio=band_ratio,
        ),
    }

    if debug:
        _generate_debug_visualisation(
            pdf_path.stem,
            page_width,
            page_height,
            labels,
            {direction: result.labels for direction, result in results.items()},
        )

    return {key: value.to_payload() for key, value in results.items()}


def _extract_labels(
    pdf_path: Path, min_font: float
) -> Tuple[List[TextLabel], float, float]:
    laparams = LAParams(detect_vertical=False, all_texts=True)
    labels: List[TextLabel] = []
    page_width = page_height = 0.0

    for page_layout in extract_pages(pdf_path, laparams=laparams):
        # ``page_layout.bbox`` gives (x0, y0, x1, y1).
        page_width = float(page_layout.bbox[2] - page_layout.bbox[0])
        page_height = float(page_layout.bbox[3] - page_layout.bbox[1])

        for element in page_layout:
            if not isinstance(element, LTTextContainer):
                continue
            for line in element:
                if isinstance(line, LTTextLine):
                    labels.extend(_process_text_line(line, min_font))

    return labels, page_width, page_height


def _process_text_line(line: LTTextLine, min_font: float) -> List[TextLabel]:
    """Split a text line into candidate labels."""

    characters: List[LTChar] = [
        obj for obj in line if isinstance(obj, LTChar) and obj.size >= min_font
    ]
    if not characters:
        return []

    segments: List[List[LTChar]] = []
    current: List[LTChar] = []

    def flush_segment() -> None:
        if not current:
            return
        text = "".join(char.get_text() for char in current)
        cleaned = _normalise_candidate_text(text)
        if cleaned is None:
            current.clear()
            return

        x0 = min(char.x0 for char in current)
        y0 = min(char.y0 for char in current)
        x1 = max(char.x1 for char in current)
        y1 = max(char.y1 for char in current)
        mean_size = statistics.mean(char.size for char in current)
        segments.append((cleaned, (x0, y0, x1, y1), mean_size))
        current.clear()

    prev_char: LTChar | None = None
    for char in characters:
        text = char.get_text()
        if text.strip() == "":
            flush_segment()
            prev_char = None
            continue

        upper = text.upper()
        if upper not in AllowedChar:
            flush_segment()
            prev_char = None
            continue

        if prev_char is not None:
            gap = char.x0 - prev_char.x1
            threshold = max(prev_char.size, char.size) * 0.8
            if gap > threshold:
                flush_segment()

        current.append(char)
        prev_char = char

    flush_segment()

    labels = [TextLabel(text, bbox, font_size) for text, bbox, font_size in segments]
    return labels


def _normalise_candidate_text(raw: str) -> str | None:
    raw = raw.strip().upper()
    if not raw:
        return None

    # Remove internal whitespace that may linger after the segmentation.
    raw = "".join(ch for ch in raw if ch in AllowedChar)
    if not raw:
        return None

    if raw.isdigit():
        value = int(raw)
        if value == 0 or value > 99:
            return None
        return str(value)

    if raw.isalpha():
        if len(raw) == 1:
            return raw
        return None

    if raw[0].isalpha() and raw[1:].isdigit():
        number = int(raw[1:])
        if number == 0 or number > 99:
            return None
        return f"{raw[0]}{number}"

    return None


def _recover_direction(
    labels: Sequence[TextLabel],
    *,
    axis: str,
    edge: str,
    page_width: float,
    page_height: float,
    margin_ratio: float,
    band_ratio: float,
) -> DirectionResult:
    assert axis in {"horizontal", "vertical"}
    assert edge in {"top", "bottom", "left", "right"}

    axis_index = 0 if axis == "vertical" else 1
    perpendicular_index = 1 - axis_index

    page_dim = page_width if axis == "vertical" else page_height

    candidates = [
        label
        for label in labels
        if _within_margin(label, edge, page_width, page_height, margin_ratio)
    ]

    if not candidates:
        candidates = list(labels)

    band_labels = _select_band(candidates, axis_index, band_ratio, page_dim)
    if not band_labels:
        band_labels = list(candidates)

    band_labels = _deduplicate_labels(band_labels, edge, perpendicular_index)

    sorted_labels = sorted(
        band_labels,
        key=lambda label: label.position[perpendicular_index],
    )

    longest_chain = _longest_consecutive_chain(sorted_labels)
    coverage = (
        len(longest_chain) / len(sorted_labels) if sorted_labels else 0.0
    )

    return DirectionResult(longest_chain, coverage)


def _within_margin(
    label: TextLabel,
    edge: str,
    page_width: float,
    page_height: float,
    margin_ratio: float,
) -> bool:
    x, y = label.position
    if edge == "top":
        return y >= page_height * (1.0 - margin_ratio)
    if edge == "bottom":
        return y <= page_height * margin_ratio
    if edge == "left":
        return x <= page_width * margin_ratio
    if edge == "right":
        return x >= page_width * (1.0 - margin_ratio)
    raise ValueError(edge)


def _select_band(
    candidates: Sequence[TextLabel],
    axis_index: int,
    band_ratio: float,
    page_dim: float,
) -> List[TextLabel]:
    if not candidates:
        return []

    if page_dim <= 0:
        return list(candidates)

    sizes = [
        (label.bbox[3] - label.bbox[1])
        if axis_index == 0
        else (label.bbox[2] - label.bbox[0])
        for label in candidates
    ]
    typical_size = statistics.median(sizes) if sizes else 0.0
    band_width = max(band_ratio * page_dim, typical_size * 3.0, 1.0)

    best_center = None
    best_count = -1

    for label in candidates:
        center = label.position[axis_index]
        count = sum(
            1
            for other in candidates
            if abs(other.position[axis_index] - center) <= band_width / 2.0
        )
        if count > best_count:
            best_center = center
            best_count = count

    if best_center is None:
        return list(candidates)

    half_band = band_width / 2.0
    return [
        label
        for label in candidates
        if abs(label.position[axis_index] - best_center) <= half_band
    ]


def _deduplicate_labels(
    labels: Sequence[TextLabel],
    edge: str,
    perpendicular_index: int,
) -> List[TextLabel]:
    result: Dict[str, TextLabel] = {}

    def is_better(new: TextLabel, existing: TextLabel) -> bool:
        if new.font_size != existing.font_size:
            return new.font_size > existing.font_size
        if edge == "top":
            return new.position[1] > existing.position[1]
        if edge == "bottom":
            return new.position[1] < existing.position[1]
        if edge == "left":
            return new.position[0] < existing.position[0]
        if edge == "right":
            return new.position[0] > existing.position[0]
        return False

    for label in labels:
        key = (label.text, round(label.position[perpendicular_index], 2))
        stored = result.get(key)
        if stored is None or is_better(label, stored):
            result[key] = label

    return list(result.values())


def _longest_consecutive_chain(labels: Sequence[TextLabel]) -> List[TextLabel]:
    if not labels:
        return []

    best: List[TextLabel] = []
    current: List[TextLabel] = []
    prev_value: int | None = None

    for label in labels:
        value = _code_value(label.text)
        if prev_value is None:
            current = [label]
        elif value - prev_value == 1:
            current.append(label)
        elif value == prev_value:
            if label.font_size > current[-1].font_size:
                current[-1] = label
        else:
            if len(current) > len(best):
                best = list(current)
            current = [label]
        prev_value = value

    if len(current) > len(best):
        best = list(current)

    return best


def _code_value(text: str) -> int:
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.isalpha():
        return 1000 + (ord(text) - ord("A")) + 1
    if len(text) >= 2 and text[0].isalpha() and text[1:].isdigit():
        letter_index = ord(text[0]) - ord("A")
        number = int(text[1:])
        return 2000 + letter_index * 100 + number
    # Fallback for unexpected input.
    return 10_000


def _generate_debug_visualisation(
    stem: str,
    width: float,
    height: float,
    labels: Sequence[TextLabel],
    recovered: Dict[str, Sequence[TextLabel]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return

    fig, ax = plt.subplots(figsize=(8, 8 * height / max(width, 1e-6)))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.scatter(
        [label.position[0] for label in labels],
        [label.position[1] for label in labels],
        s=10,
        c="lightgray",
        label="candidates",
    )

    colours = {
        "top": "tab:red",
        "bottom": "tab:blue",
        "left": "tab:green",
        "right": "tab:purple",
    }

    for direction, direction_labels in recovered.items():
        if not direction_labels:
            continue
        ax.scatter(
            [label.position[0] for label in direction_labels],
            [label.position[1] for label in direction_labels],
            s=40,
            c=colours.get(direction, "black"),
            label=direction,
        )

    ax.legend()
    fig.tight_layout()
    output_path = Path(f"{stem}_debug_markers.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


__all__ = ["extract_zone_markers", "TextLabel", "DirectionResult"]

