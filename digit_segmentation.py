import cv2
import imutils

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

def digit_recognition(thresh_img):
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
    digitCnts = []

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
            digitCnts.append(c)

    digitCnts = sorted(digitCnts, key=lambda c: cv2.boundingRect(c)[0])
    return digitCnts
