import cv2
from pathlib import Path
from roi_detection import preprocess, find_big_displays, crop_from_original
from digit_segmentation import read_display, thresholding_digit


if __name__ == "__main__":
    image_path = Path(__file__).resolve().parent / "img" / "raw" / "raw_009.jpg"
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    resized, gray, edged = preprocess(img)
    displays = find_big_displays(edged)
    rois = crop_from_original(img, resized, displays, pad=5)

    vis = resized.copy()

    for i, (x, y, w, h, area) in enumerate(displays):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i, (x0, y0, x1, y1, roi) in enumerate(rois):
        thresh = thresholding_digit(roi)
        result, components = read_display(thresh)

        roi_vis = roi.copy()
        print(f"roi_{i}: {result}")

        for component in components:
            x, y, w, h = component["bbox"]
            color = (0, 255, 255) if component["kind"] == "decimal" else (0, 255, 0)
            cv2.rectangle(roi_vis, (x, y), (x + w, y + h), color, 3)

            label = component.get("value", component["kind"])
            cv2.putText(
                roi_vis,
                label,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            roi_vis,
            result,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.imshow(f"roi_{i}", roi_vis)
        cv2.imshow(f"thresh_{i}", thresh)

    cv2.imshow("detected", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
