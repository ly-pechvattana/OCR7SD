import cv2
from roi_detection import preprocess, find_big_displays, crop_from_original
from digit_segmentation import thresholding_digit, digit_recognition


if __name__ == "__main__":
    img = cv2.imread("./img/raw/raw_012.jpg")

    resized, gray, edged = preprocess(img)
    displays = find_big_displays(edged)
    rois = crop_from_original(img, resized, displays, pad=5)

    vis = resized.copy()

    for i, (x, y, w, h, area) in enumerate(displays):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i, (x0, y0, x1, y1, roi) in enumerate(rois):
        thresh = thresholding_digit(roi)
        digitCnts = digit_recognition(thresh)
        
        # Draw contours on the ROI
        roi_vis = roi.copy()
        cv2.drawContours(roi_vis, digitCnts, -1, (0, 255, 0), 2)
        
        cv2.imshow(f"roi_{i}", roi_vis)

    #cv2.imshow("edged", edged)
    cv2.imshow("detected", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
