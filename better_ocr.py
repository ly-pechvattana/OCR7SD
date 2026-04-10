
import cv2
import imutils

def cnvt_edged_image(img_arr, should_save=False):
  # ratio = img_arr.shape[0] / 300.0
  image = imutils.resize(img_arr,height=300)
  gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),11, 17, 17)
  edged_image = cv2.Canny(gray_image, 20, 100)

  if should_save:
    cv2.imwrite('cntr_ocr.jpg', edged_image)

  return edged_image

if __name__ == "__main__":
    img_arr = cv2.imread('./img/raw/raw_004.jpg')
    edged_image = cnvt_edged_image(img_arr, should_save=True)
    cv2.imshow('edged_image', edged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()