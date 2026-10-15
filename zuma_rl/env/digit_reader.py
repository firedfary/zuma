import os
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

class ZumaDigitReader:
    """
    Recognizes golden score digits in Zuma's top bar using template matching
    against PopCap's official Cancun13 bitmap font.
    """
    def __init__(self, font_path: Optional[str] = None):
        if font_path is None:
            font_path = r"d:\zuma\祖玛豪华版\fonts\_Cancun13.gif"
        
        self.templates: Dict[int, np.ndarray] = {}
        self._load_templates(font_path)

    def _load_templates(self, font_path: str):
        # Cancun13 rect list from cancun13.txt:
        rects = [
            (0, 0, 12, 14),   # '0'
            (13, 0, 5, 14),   # '1'
            (19, 0, 11, 14),  # '2'
            (31, 0, 9, 14),   # '3'
            (41, 0, 12, 14),  # '4'
            (54, 0, 9, 14),   # '5'
            (64, 0, 10, 14),  # '6'
            (75, 0, 10, 14),  # '7'
            (86, 0, 10, 14),  # '8'
            (97, 0, 10, 14),  # '9'
        ]

        if os.path.exists(font_path):
            try:
                with Image.open(font_path) as im:
                    gray = np.array(im.convert("L"))
                for digit, (x, y, w, h) in enumerate(rects):
                    tmpl = gray[y:y+h, x:x+w]
                    # Binarize or normalize
                    _, tmpl_bin = cv2.threshold(tmpl, 100, 255, cv2.THRESH_BINARY)
                    self.templates[digit] = tmpl_bin
                return
            except Exception as e:
                print(f"[digit_reader] Failed to load font from {font_path}: {e}")

        # Fallback: synthesize default 14px digit templates using OpenCV
        print("[digit_reader] Synthesizing default digit templates as fallback.")
        for d in range(10):
            canvas = np.zeros((14, 11), dtype=np.uint8)
            cv2.putText(canvas, str(d), (1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 255, 1, cv2.LINE_AA)
            self.templates[d] = canvas

    def extract_score(self, frame: np.ndarray, roi: Tuple[int, int, int, int] = (2, 22, 265, 420)) -> int:
        """
        Extract the integer score from the score ROI.
        roi is (y_start, y_end, x_start, x_end) in 640x480 frame.
        """
        y1, y2, x1, x2 = roi
        if frame is None or frame.shape[0] < y2 or frame.shape[1] < x2:
            return 0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0

        # Golden text filter
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        # Gold hue is around 10-40, high saturation, high value
        mask = cv2.inRange(hsv, np.array([10, 70, 100]), np.array([45, 255, 255]))
        
        # If no gold pixels, check fallback grayscale thresholding
        if np.sum(mask > 0) < 15:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        matches: List[Tuple[int, int, float, int]] = []  # (x, digit, score, width)

        for digit, tmpl in self.templates.items():
            th, tw = tmpl.shape
            if mask.shape[0] < th or mask.shape[1] < tw:
                continue
            res = cv2.matchTemplate(mask, tmpl, cv2.TM_CCOEFF_NORMED)
            threshold = 0.55
            locs = np.where(res >= threshold)
            for pt in zip(*locs[::-1]):
                val = float(res[pt[1], pt[0]])
                matches.append((pt[0], digit, val, tw))

        if not matches:
            return 0

        # Proper Non-Maximum Suppression (NMS):
        # 1. Sort matches by confidence score descending to prioritize high-confidence detections
        matches.sort(key=lambda m: m[2], reverse=True)

        # 2. Greedily accept highest-confidence detection and suppress overlapping candidates
        # using 1D bounding box interval intersection
        filtered: List[Tuple[int, int, float, int]] = []
        for m in matches:
            x, digit, val, tw = m
            overlap = False
            for fx, fdigit, fval, ftw in filtered:
                inter = max(0, min(x + tw, fx + ftw) - max(x, fx))
                if inter / min(tw, ftw) > 0.4:
                    overlap = True
                    break
            if not overlap:
                filtered.append(m)

        if not filtered:
            return 0

        # 3. Sort surviving detections left-to-right by x coordinate
        filtered.sort(key=lambda item: item[0])

        # 4. Concatenate digits into integer
        score_str = "".join([str(d) for _, d, _, _ in filtered])
        try:
            return int(score_str)
        except ValueError:
            return 0
