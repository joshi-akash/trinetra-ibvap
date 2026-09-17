"""
Automatic Number Plate Recognition (ANPR) OCR for TRINETRA.

Runs OCR (PaddleOCR primary, EasyOCR fallback) on vehicle/plate crops,
enforces Indian license plate regex: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$,
and penalizes confidence if format validation fails.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard Indian License Plate Regex:
# e.g., DL01A1234, HR26DQ5551, UP32AB1234, MH04F1122, JK02C9988
INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$")

# Common OCR confusion character mapping
CHAR_REPLACEMENTS = {
    " ": "",
    "-": "",
    ".": "",
    "_": "",
    ":": "",
    "/": "",
    "|": "",
}


def clean_plate_text(raw_text: str) -> str:
    """
    Clean and canonicalize raw OCR text.
    Removes whitespace, punctuation, and converts to uppercase alphanumeric.
    """
    if not raw_text:
        return ""

    text = raw_text.upper().strip()
    for char, rep in CHAR_REPLACEMENTS.items():
        text = text.replace(char, rep)

    # Filter strictly to alphanumeric characters
    cleaned = re.sub(r"[^A-Z0-9]", "", text)
    return cleaned


def validate_indian_plate(plate_text: str) -> bool:
    """
    Validate whether the cleaned plate text conforms to standard Indian vehicle registration regex:
    ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$
    """
    if not plate_text:
        return False
    return bool(INDIAN_PLATE_PATTERN.match(plate_text))


class PlateOCR:
    """
    Plate OCR engine with Indian format validation and confidence penalization.
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
        """Initialize PaddleOCR, then EasyOCR fallback."""
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

    def read_plate(self, plate_crop: np.ndarray) -> Optional[Tuple[str, float, bool]]:
        """
        Extract text from a cropped plate image.

        Args:
            plate_crop: np.ndarray image patch

        Returns:
            Tuple of (cleaned_text, confidence_score, is_valid_format),
            or None if no text could be recognized.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        h, w = plate_crop.shape[:2]
        if h < 10 or w < 20:
            return None

        # Preprocessing: upscale small crops and enhance contrast for sharp OCR
        proc_crop = plate_crop
        try:
            import cv2
            if h < 48:
                scale_f = 48.0 / max(1.0, float(h))
                new_w = max(40, int(w * scale_f))
                proc_crop = cv2.resize(plate_crop, (new_w, 48), interpolation=cv2.INTER_CUBIC)
            
            # Contrast enhancement
            if len(proc_crop.shape) == 3 and proc_crop.shape[2] == 3:
                lab = cv2.cvtColor(proc_crop, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                cl = clahe.apply(l)
                proc_crop = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:
            proc_crop = plate_crop

        extracted_text = ""
        raw_conf = 0.0

        if self._engine_type == "paddle" and self.ocr_engine is not None:
            try:
                result = self.ocr_engine.ocr(proc_crop, cls=self.use_angle_cls)
                if result and len(result) > 0 and result[0]:
                    texts = []
                    confs = []
                    for line in result[0]:
                        txt = line[1][0]
                        score = float(line[1][1])
                        texts.append(txt)
                        confs.append(score)
                    extracted_text = "".join(texts)
                    raw_conf = float(np.mean(confs)) if confs else 0.0
            except Exception as e:
                logger.error("PaddleOCR inference error: %s", e)

        elif self._engine_type == "easyocr" and self.ocr_engine is not None:
            try:
                result = self.ocr_engine.readtext(proc_crop)
                if result:
                    texts = [r[1] for r in result]
                    confs = [float(r[2]) for r in result]
                    extracted_text = "".join(texts)
                    raw_conf = float(np.mean(confs)) if confs else 0.0
            except Exception as e:
                logger.error("EasyOCR inference error: %s", e)

        if not extracted_text:
            return None

        cleaned = clean_plate_text(extracted_text)
        if not cleaned or len(cleaned) < 4:
            return None

        # Enforce Indian plate regex validation
        is_valid = validate_indian_plate(cleaned)

        if is_valid:
            final_conf = raw_conf
        else:
            # Penalize confidence if formatting fails (FR-ANPR-02)
            final_conf = raw_conf * self.confidence_penalty_factor
            logger.debug(
                "Plate '%s' failed Indian format regex. Confidence penalized: %.2f -> %.2f",
                cleaned,
                raw_conf,
                final_conf,
            )

        return (cleaned, round(final_conf, 4), is_valid)
