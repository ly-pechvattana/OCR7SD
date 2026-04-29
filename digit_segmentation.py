import cv2
import imutils
import numpy as np


DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): "0",
    (0, 0, 1, 0, 0, 1, 0): "1",
    (1, 0, 1, 1, 1, 0, 1): "2",
    (1, 0, 1, 1, 0, 1, 1): "3",
    (0, 1, 1, 1, 0, 1, 0): "4",
    (1, 1, 0, 1, 0, 1, 1): "5",
    (1, 1, 0, 1, 1, 1, 1): "6",
    (1, 0, 1, 0, 0, 1, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9",
}

def thresholding_digit(roi_img):
    b_channel, g_channel, r_channel = cv2.split(roi_img)

    # Seven-segment LEDs are strongly red, so isolate that signal instead of
    # inverting a grayscale image, which turns the whole dark display on.
    red_score = cv2.subtract(r_channel, cv2.max(b_channel, g_channel))
    red_score = cv2.GaussianBlur(red_score, (5, 5), 0)

    thresh = cv2.threshold(
        red_score,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )[1]

    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, small_kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, small_kernel)

    return thresh

def _extract_components(thresh_img):
    roi_h, roi_w = thresh_img.shape[:2]

    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(25, roi_w // 80),
            max(25, roi_h // 24),
        ),
    )
    merged = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, merge_kernel)

    cnts = cv2.findContours(merged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    components = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        area = w * h
        is_digit = h >= int(roi_h * 0.45) and w >= int(roi_w * 0.04)
        is_decimal = (
            area >= int(roi_h * roi_w * 0.0015)
            and 0.6 <= (w / float(h)) <= 1.4
            and y >= int(roi_h * 0.65)
        )

        if is_digit or is_decimal:
            components.append(
                {
                    "bbox": (x, y, w, h),
                    "kind": "decimal" if is_decimal and not is_digit else "digit",
                }
            )

    components.sort(key=lambda item: item["bbox"][0])
    return components


def _normalize_digit_crop(binary_crop):
    ys, xs = np.where(binary_crop > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    tight = binary_crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(tight, (100, 180), interpolation=cv2.INTER_NEAREST)


def decode_digit(binary_crop):
    normalized = _normalize_digit_crop(binary_crop)
    if normalized is None:
        return "?", (0, 0, 0, 0, 0, 0, 0), [0.0] * 7

    segments = [
        ((22, 8), (78, 30)),
        ((6, 22), (28, 82)),
        ((72, 22), (94, 82)),
        ((22, 76), (78, 104)),
        ((6, 98), (28, 158)),
        ((72, 98), (94, 158)),
        ((22, 150), (78, 172)),
    ]

    pattern = []
    fill_ratios = []

    for (x0, y0), (x1, y1) in segments:
        segment_roi = normalized[y0:y1, x0:x1]
        fill_ratio = cv2.countNonZero(segment_roi) / float(segment_roi.size)
        fill_ratios.append(fill_ratio)
        pattern.append(1 if fill_ratio > 0.35 else 0)

    pattern = tuple(pattern)
    value = DIGITS_LOOKUP.get(pattern, "?")
    rounded_ratios = [round(ratio, 3) for ratio in fill_ratios]
    return value, pattern, rounded_ratios


def read_display(thresh_img):
    components = _extract_components(thresh_img)
    chars = []

    for component in components:
        x, y, w, h = component["bbox"]

        if component["kind"] == "decimal":
            component["value"] = "."
            chars.append(".")
            continue

        digit_crop = thresh_img[y:y + h, x:x + w]
        value, pattern, fill_ratios = decode_digit(digit_crop)
        component["value"] = value
        component["pattern"] = pattern
        component["fill_ratios"] = fill_ratios
        chars.append(value)

    return "".join(chars), components


def digit_recognition(thresh_img):
    components = _extract_components(thresh_img)
    digitCnts = []

    for component in components:
        x, y, w, h = component["bbox"]
        contour = np.array(
            [
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]],
            ],
            dtype=np.int32,
        )
        digitCnts.append(contour)

    return digitCnts
