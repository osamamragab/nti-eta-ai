import cv2


img = cv2.imread("lol.jpg")
cv2.imshow("img", img)
cv2.waitKey(500)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", img_gray)
cv2.waitKey(500)

cv2.destroyAllWindows()

for t in (
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV,
    cv2.THRESH_TRUNC,
):
    print(t)
    (thresh, img_bw) = cv2.threshold(img_gray, 200, 255, t)
    cv2.imshow("img", img_bw)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
