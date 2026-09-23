"""
Automatic Number Plate Recognition (ANPR) OCR for TRINETRA.

Runs multi-engine / multi-pass OCR (PaddleOCR primary, EasyOCR with GPU acceleration)
with specialized HSRP (High Security Registration Plate) preprocessing:
- Supports standard Indian plates: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$ (e.g. KA02MN1826, DL8CAF5030)
- Supports Bharat (BH) series: ^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$ (e.g. 22BH1234AA)
- Automatic HSRP "IND" badge stripping
- Positional character error correction (disambiguates O/0, I/1, Z/2, B/8, S/5, A/4)
- Multi-pass enhancement (CLAHE, Otsu binarization, inverted retroreflective, unsharp mask)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard Indian License Plate Regex:
# e.g., DL01A1234, HR26DQ5551, UP32AB1234, MH04F1122, KA02MN1826, DL8CAF5030
INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BH_PLATE_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Common OCR confusion character mapping
CHAR_REPLACEMENTS = {
    " ": "",
    "-": "",
    ".": "",
    "_": "",
    ":": "",
    "/": "",
    "|": "",
    ",": "",
    ";": "",
}

# Positional disambiguation tables
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8", "D": "0", "Q": "0"}


def correct_plate_characters(text: str) -> str:
    """
    Applies standard positional error correction for Indian license plates.
    Disambiguates OCR confusions (O/0, I/1, Z/2, B/8, S/5).
    """
    if not text or len(text) not in (9, 10, 11):
        return text

    chars = list(text)
    n = len(chars)

    # State code (First 2 chars): ALWAYS LETTERS
    for i in (0, 1):
        if chars[i] in DIGIT_TO_LETTER:
            chars[i] = DIGIT_TO_LETTER[chars[i]]

    # District code & Series & Vehicle number
    if n == 9:
        # Standard Indian 9-char format: SS DD L NNNN (e.g., DL 01 A 1234, JK 02 C 9988)
        for i in (2, 3):
            if chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]
        if chars[4] in DIGIT_TO_LETTER:
            chars[4] = DIGIT_TO_LETTER[chars[4]]
        for i in range(5, 9):
            if chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]
    elif n == 10:
        # e.g., KA 02 MN 1826
        for i in (2, 3):
            if chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]
        for i in (4, 5):
            if chars[i] in DIGIT_TO_LETTER:
                chars[i] = DIGIT_TO_LETTER[chars[i]]
        for i in range(6, 10):
            if chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]
    elif n == 11:
        # e.g., DL 08 CAF 5030
        for i in (2, 3):
            if chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]
        for i in range(4, 7):
            if chars[i] in DIGIT_TO_LETTER:
                chars[i] = DIGIT_TO_LETTER[chars[i]]
        for i in range(7, 11):
            if chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]

    return "".join(chars)


def clean_plate_text(raw_text: str) -> str:
    """
    Clean, canonicalize, and correct raw OCR text.
    Removes punctuation, strips HSRP 'IND' badges, and applies positional correction.
    """
    if not raw_text:
        return ""

    text = raw_text.upper().strip()
    for char, rep in CHAR_REPLACEMENTS.items():
        text = text.replace(char, rep)

    # Filter strictly to alphanumeric characters
    cleaned = re.sub(r"[^A-Z0-9]", "", text)

    # Strip HSRP "IND" or "1ND" leading badge
    if cleaned.startswith("IND") and len(cleaned) >= 9:
        sub = cleaned[3:]
        if validate_indian_plate(sub) or len(sub) in (9, 10, 11):
            cleaned = sub
    elif cleaned.startswith("1ND") and len(cleaned) >= 9:
        sub = cleaned[3:]
        if validate_indian_plate(sub) or len(sub) in (9, 10, 11):
            cleaned = sub

    # Apply positional error correction
    cleaned = correct_plate_characters(cleaned)
    return cleaned


def validate_indian_plate(plate_text: str) -> bool:
    """
    Validate whether the cleaned plate text conforms to Indian vehicle registration standards:
    - Standard: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$
    - Bharat Series: ^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$
    """
    if not plate_text:
        return False
    if INDIAN_PLATE_PATTERN.match(plate_text):
        return True
    if BH_PLATE_PATTERN.match(plate_text):
        return True
    return False


class PlateOCR:
    """
    Multi-Engine / Multi-Pass Plate OCR with Indian HSRP format validation
    and positional confidence recovery.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        confidence_penalty_factor: float = 0.5,
    ):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.confidence_penalty_factor = confidence_penalty_factor
        self.ocr_engine = None
        self._engine_type = "none"

        self._init_ocr()

    def _init_ocr(self) -> None:
        """Initialize EasyOCR (GPU) or PaddleOCR fallback."""
        try:
            from paddleocr import PaddleOCR as PD_OCR
            self.ocr_engine = PD_OCR(use_angle_cls=self.use_angle_cls, lang=self.lang, show_log=False)
            self._engine_type = "paddle"
            logger.info("PaddleOCR initialized successfully for ANPR.")
            return
        except Exception as e:
            logger.warning("PaddleOCR not available (%s). Trying EasyOCR fallback.", e)

        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            self.ocr_engine = easyocr.Reader([self.lang], gpu=use_gpu, verbose=False)
            self._engine_type = "easyocr"
            logger.info("EasyOCR initialized successfully as ANPR engine (GPU=%s).", use_gpu)
            return
        except Exception as e:
            logger.warning("EasyOCR not available (%s). Operating in fallback mode.", e)

        self.ocr_engine = None
        self._engine_type = "none"

    def _run_raw_ocr(self, img_patch: np.ndarray) -> List[Tuple[str, float]]:
        """Executes the active OCR engine on an image patch."""
        segments: List[Tuple[str, float]] = []
        if self._engine_type == "easyocr" and self.ocr_engine is not None:
            try:
                res = self.ocr_engine.readtext(img_patch)
                if res:
                    for r in res:
                        segments.append((str(r[1]), float(r[2])))
            except Exception:
                pass
        elif self._engine_type == "paddle" and self.ocr_engine is not None:
            try:
                res = self.ocr_engine.ocr(img_patch, cls=self.use_angle_cls)
                if res and len(res) > 0 and res[0]:
                    for line in res[0]:
                        segments.append((str(line[1][0]), float(line[1][1])))
            except Exception:
                pass
        return segments

    def read_candidate_plates(self, image_patch: np.ndarray) -> List[Tuple[str, float, bool]]:
        """
        Extract all candidate license plate strings from an image patch with multi-pass HSRP enhancement.
        Evaluates normal CLAHE, Otsu binarization, and inverted retroreflective passes.
        """
        if image_patch is None or image_patch.size == 0:
            return []

        h, w = image_patch.shape[:2]
        if h < 10 or w < 20:
            return []

        # Build Multi-Pass Image Enhancements
        passes: List[np.ndarray] = []

        # Pass 1: Upscale and LAB CLAHE (Standard contrast boost)
        proc_crop = image_patch
        try:
            if h < 52:
                scale_f = 52.0 / max(1.0, float(h))
                new_w = max(45, int(w * scale_f))
                proc_crop = cv2.resize(image_patch, (new_w, 52), interpolation=cv2.INTER_CUBIC)

            if len(proc_crop.shape) == 3 and proc_crop.shape[2] == 3:
                lab = cv2.cvtColor(proc_crop, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
                cl = clahe.apply(l)
                proc_crop = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            passes.append(proc_crop)
        except Exception:
            passes.append(image_patch)

        # Pass 2: Otsu Adaptive Binarization (Embossed black-on-white HSRP characters)
        try:
            gray = cv2.cvtColor(proc_crop, cv2.COLOR_BGR2GRAY) if len(proc_crop.shape) == 3 else proc_crop
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            passes.append(otsu)
        except Exception:
            pass

        # Pass 3: Inverted Binarization (Retroreflective plates or nighttime headlights glare)
        try:
            if len(passes) >= 2:
                inv_otsu = cv2.bitwise_not(passes[1])
                passes.append(inv_otsu)
        except Exception:
            pass

        candidates: List[Tuple[str, float, bool]] = []
        seen_texts = set()

        for pass_img in passes:
            raw_segments = self._run_raw_ocr(pass_img)
            if not raw_segments:
                continue

            # 1. Individual text segments (e.g. ['KA02MN1826'])
            for txt, conf in raw_segments:
                cleaned = clean_plate_text(txt)
                if cleaned and len(cleaned) >= 6 and cleaned not in seen_texts:
                    is_valid = validate_indian_plate(cleaned)
                    if is_valid:
                        candidates.append((cleaned, round(conf, 4), True))
                        seen_texts.add(cleaned)

            # 2. Pairwise adjacent joined segments (e.g. ['DL01', 'AB1234'])
            for i in range(len(raw_segments) - 1):
                pair_raw = raw_segments[i][0] + raw_segments[i + 1][0]
                cleaned_pair = clean_plate_text(pair_raw)
                if cleaned_pair and len(cleaned_pair) >= 6 and cleaned_pair not in seen_texts:
                    is_valid = validate_indian_plate(cleaned_pair)
                    if is_valid:
                        avg_conf = (raw_segments[i][1] + raw_segments[i + 1][1]) / 2.0
                        candidates.append((cleaned_pair, round(avg_conf, 4), True))
                        seen_texts.add(cleaned_pair)

            # 3. All joined segments
            all_raw = "".join([s[0] for s in raw_segments])
            cleaned_all = clean_plate_text(all_raw)
            if cleaned_all and len(cleaned_all) >= 6 and cleaned_all not in seen_texts:
                is_valid = validate_indian_plate(cleaned_all)
                avg_conf = sum(s[1] for s in raw_segments) / len(raw_segments)
                if is_valid:
                    candidates.append((cleaned_all, round(avg_conf, 4), True))
                    seen_texts.add(cleaned_all)
                elif not candidates:
                    penalized_conf = avg_conf * self.confidence_penalty_factor
                    candidates.append((cleaned_all, round(penalized_conf, 4), False))

            # If we found a valid plate in this pass, no need to run further passes
            if any(c[2] for c in candidates):
                break

        # Sort: Valid plates first, then by confidence descending
        candidates.sort(key=lambda x: (1 if x[2] else 0, x[1]), reverse=True)
        return candidates

    def read_plate(self, plate_crop: np.ndarray) -> Optional[Tuple[str, float, bool]]:
        """Extract text from a cropped plate image."""
        candidates = self.read_candidate_plates(plate_crop)
        if candidates:
            return candidates[0]
        return None
