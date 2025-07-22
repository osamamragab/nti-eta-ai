import cv2 as cv

img = cv.imread("corners.jpg")
assert img is not None
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
cv.waitKey(0)
cv.destroyAllWindows()

harris = cv.cornerHarris(gray, 2, 3, 0.04)
harris_dst = cv.dilate(harris, None)
harris_img = img.copy()
harris_mask = harris_dst > 0.01 * harris_dst.max()
harris_img[harris_mask] = [0, 0, 255]
cv.imshow("harris", harris_img)
cv.waitKey(0)
cv.destroyAllWindows()
