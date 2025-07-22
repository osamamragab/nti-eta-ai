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

# canny edge
canny = cv.Canny(gray, 50, 150)
cv.imshow("canny", canny)
cv.waitKey(0)
cv.destroyAllWindows()

# sobel edge
sobel = cv.Sobel(gray, cv.CV_64F, 1, 1, ksize=3)
cv.imshow("sobel", sobel)
cv.waitKey(0)
cv.destroyAllWindows()

# laplacian edge
laplacian = cv.Laplacian(gray, cv.CV_64F)
# laplacian = np.uint8(np.absolute(laplacian))
cv.imshow("laplacian", laplacian)
cv.waitKey(0)
cv.destroyAllWindows()

# harris corner
harris = cv.cornerHarris(gray, 2, 3, 0.04)
harris_dst = cv.dilate(harris, None)
harris_img = img.copy()
harris_mask = harris_dst > 0.01 * harris_dst.max()
harris_img[harris_mask] = [0, 0, 255]
cv.imshow("harris", harris_img)
cv.waitKey(0)
cv.destroyAllWindows()

# sift corner
sift = cv.SIFT_create()
sift_kp, _ = sift.detectAndCompute(gray, None)
sift_img = img.copy()
sift_img = cv.drawKeypoints(sift_img, sift_kp, sift_img, color=(0, 255, 0))
cv.imshow("sift", sift_img)
cv.waitKey(0)
cv.destroyAllWindows()

# fast corner
fast = cv.FastFeatureDetector_create()
fast_kp = fast.detect(gray, None)
fast_img = img.copy()
img_fast = cv.drawKeypoints(fast_img, fast_kp, fast_img, color=(0, 0, 255))
cv.imshow("fast", img_fast)
cv.waitKey(0)
cv.destroyAllWindows()

# Print summary
print("EDGE DETECTORS:")
print("- Canny: Best overall edge detection, good noise suppression")
print("- Sobel: Good for detecting edges in specific directions")
print("- Laplacian: Sensitive to noise, detects edges in all directions")
print()
print("CORNER DETECTORS:")
print("- Harris: Classic corner detector, rotation invariant")
print("- Shi-Tomasi: Improved version of Harris, better corner selection")
print("- FAST: Very fast, good for real-time applications")
print()
print("Key Differences:")
print("- Edges: Continuous boundaries, many pixels")
print("- Corners: Discrete points, fewer but more stable features")
