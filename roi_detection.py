import cv2
import imutils

def preprocess(img):
    resized = imutils.resize(img, height=300)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 30, 150)
    edged = cv2.morphologyEx(
        edged,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2
    )

    return resized, gray, edged

def find_big_displays(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    candidates = []

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect = w / float(h)

        # tune these values for your camera
        if w > 180 and h > 50 and area > 8000 and aspect > 3.0:
            candidates.append((x, y, w, h, area))

    # keep only the 2 biggest displays
    candidates = sorted(candidates, key=lambda item: item[4], reverse=True)[:2]

    # sort top-to-bottom
    candidates = sorted(candidates, key=lambda item: item[1])

    return candidates

def crop_from_original(original, resized, candidates, pad=5):
    scale_x = original.shape[1] / float(resized.shape[1])
    scale_y = original.shape[0] / float(resized.shape[0])

    rois = []

    for (x, y, w, h, area) in candidates:
        x0 = int((x + pad) * scale_x)
        y0 = int((y + pad) * scale_y)
        x1 = int((x + w - pad) * scale_x)
        y1 = int((y + h - pad) * scale_y)

        roi = original[y0:y1, x0:x1]
        rois.append((x0, y0, x1, y1, roi))

    return rois

if __name__ == "__main__":
    img = cv2.imread("./img/raw/raw_015.jpg")

    resized, gray, edged = preprocess(img)
    displays = find_big_displays(edged)
    rois = crop_from_original(img, resized, displays, pad=5)

    vis = resized.copy()

    for i, (x, y, w, h, area) in enumerate(displays):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i, (x0, y0, x1, y1, roi) in enumerate(rois):
        cv2.imshow(f"roi_{i}", roi)

    #cv2.imshow("edged", edged)
    cv2.imshow("detected", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()