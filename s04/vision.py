import cv2


# img = cv2.imread("lol.jpg")
# cv2.imshow("img", img)
# cv2.waitKey(500)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

# img = cv2.imread("lol.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("img", img)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
#
# img = cv2.imread("lol.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_8)
# cv2.imshow("img", img)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

# cv2.imwrite("fuckyeah.jpg", img)

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", img_gray)

# (thresh, img_bw) = cv2.threshold(img_gray, 127,255, cv2.THRESH_BINARY)
# cv2.imshow("bw", img_bw)

# cv2.waitKey(500)
# cv2.destroyAllWindows()

# B, G, R = cv2.split(img)
# cv2.imshow("B", B)
# cv2.waitKey(500)
# cv2.imshow("G", G)
# cv2.waitKey(500)
# cv2.imshow("R", R)
# cv2.waitKey(500)

# m = cv2.merge((B, G, R))
# cv2.imshow("merged", m)
# cv2.waitKey(500)
# cv2.destroyAllWindows()

# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv", img_hsv)
# cv2.waitKey(500)
#
# H = img_hsv[:,:,0]
# S = img_hsv[:,:,1]
# V = img_hsv[:,:,2]
# cv2.imshow("Hue", H)
# cv2.waitKey(500)
# cv2.imshow("Saturation", S)
# cv2.waitKey(500)
# cv2.imshow("Value", V)
# cv2.waitKey(500)
# cv2.destroyAllWindows()

# half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# big = cv2.resize(img, (1050, 1610))
# stretch = cv2.resize(img, (780, 540), interpolation=cv2.INTER_NEAREST)
#
# for i in (half, big, stretch):
#     cv2.imshow("img", i)
#     cv2.waitKey(500)
#     cv2.destroyAllWindows()

# img_blur = cv2.blur(img, (5, 5))
# cv2.imshow("average image", img_blur)
# cv2.waitKey(500)
# cv2.destroyAllWindows()
#
# img_gauss = cv2.GaussianBlur(img, (7, 7), 0)
# cv2.imshow("gaussian blur", img_gauss)
# cv2.waitKey(500)
# cv2.destroyAllWindows()
#
# img_median = cv2.medianBlur(img, 5)
# cv2.imshow("median blur", img_median)
# cv2.waitKey(500)
# cv2.destroyAllWindows()
#
# img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
# cv2.imshow("bilateral filter", img_bilateral)
# cv2.waitKey(500)
# cv2.destroyAllWindows()

# img = cv2.imread("lol.jpg", 0)
# cv2.imshow("img", img)
# cv2.waitKey(500)

# equ = cv2.equalizeHist(img)
# cv2.imshow("equ", equ)
# cv2.waitKey(500)
# cv2.destroyAllWindows()

img = cv2.imread("lol.jpg", 0)
img_canny = cv2.Canny(img, 200, 250)
cv2.imwrite("canny.jpg", img_canny)
cv2.imshow("original", img)
cv2.waitKey(500)
cv2.imshow("canny", img_canny)
cv2.waitKey(500)
cv2.destroyAllWindows()
