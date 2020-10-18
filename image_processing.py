import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def plot_two_images(img1, img2, title):
    plt.subplot(121), plt.imshow(img1, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


def show_image(img, title = ''):
    plt.imshow(img, cmap='gray')
    plt.show()


input_img = cv2.imread(f'data/image_test.png')
# images = []
# titles = []
# images.append(input_img)
# titles.append("Input img")

# Morphology

# for x in [3,5,7,9,11,13,15,17,19,21,23,25]:
kernel = np.ones((21, 21), np.uint8)
morph_img = cv2.dilate(input_img, kernel, iterations=1)
morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel)
morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((20, 20), np.uint8)
morph_img = cv2.dilate(morph_img, kernel, iterations=1)

cv2.imwrite(f'data/img_test_morph.png', morph_img)

# img_blur = cv2.medianBlur(morph_img, 3)

# cv2.imwrite(f'data/img_blur.png', img_blur)
#images.append(morph_img)
#titles.append("Img after morph")


'''
images = []
for x in [3, 5, 7, 9, 11, 13]:
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, kernel)
    images.append(closing)
show_images(images)
'''

# plot_two_images(input_img, erosion, 'Erosion')
# plot_two_images(input_img, dilation, 'Dilation')
# plot_two_images(input_img, opening, 'Opening')
# plot_two_images(input_img, closing, 'Closing')


# play with different values for threshold1 and threshold2


'''
running canny with variing threshold values --> didnt change anything...
for i in np.arange(200, 1000, 100):
    canny = cv2.Canny(closing, 200, i)
    cv2.imwrite(f'data/canny/img_canny_{i}.png', canny)
'''

'''
for x in [3,5,7]:
    canny = cv2.Canny(closing, 200, 400, apertureSize=x, L2gradient=True)
    cv2.imwrite(f'data/canny_apertureSize_{x}_L2gradient.png', canny)
'''

# edge detection
canny_img = cv2.Canny(morph_img, 75, 250, apertureSize=3, L2gradient=True)
cv2.imwrite(f'data/img_canny.png', canny_img)

exit(0)

gftt_img = np.zeros((canny_img.shape[0], canny_img.shape[1]), np.uint8)
corners = cv2.goodFeaturesToTrack(canny_img, 25, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(gftt_img, (x,y), 3, 255, -1)

cv2.imwrite(f'data/img_gftt.png', canny_img)

# Harris Corner Detector

thresh = 255
# Detector parameters
blockSize = 2
apertureSize = 3
k = 0.04
img_harris = cv2.cornerHarris(canny_img, blockSize, apertureSize, k)
# cv2.imwrite(f'data/img_harris.png', img_harris)

# normalize
img_harris_norm = np.empty(img_harris.shape, dtype=np.float32)
cv2.normalize(img_harris, img_harris_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
img_harris_scaled = cv2.convertScaleAbs(img_harris_norm)
cv2.imwrite(f'data/img_harris_norm.png', img_harris_scaled)

# Drawing a circle around corners
print(img_harris.shape)
for i in range(1000):
    for j in range(100):
        if int(img_harris[i, j]) > thresh:
            cv2.circle(img_harris, (j, i), 10, 255, 2)

cv2.imwrite(f'data/img_harris_drawn.png', img_harris)



exit(0)
# contouring
'''
contour_img = np.zeros((canny_img.shape[0], canny_img.shape[1]), np.uint8)
contour_img_2 = np.zeros((canny_img.shape[0], canny_img.shape[1]), np.uint8)



  cv::RETR_EXTERNAL = 0,
  cv::RETR_LIST = 1,
  cv::RETR_CCOMP = 2,
  cv::RETR_TREE = 3,
  cv::RETR_FLOODFILL = 4 
  
  cv::CHAIN_APPROX_NONE = 1,
  cv::CHAIN_APPROX_SIMPLE = 2,
  cv::CHAIN_APPROX_TC89_L1 = 3,
  cv::CHAIN_APPROX_TC89_KCOS = 4 

contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_img, contours, -1, 255, 3)
cv2.imwrite(f'data/contours/img_contours.png', contour_img)

histo_data = []
contours_remove = []
print(len(contours))
for i in range(len(contours)):
    contourArea = cv2.contourArea(contours[i])
    histo_data.append(contourArea)
    if contourArea < 20:
        contours_remove.append(i)

for i in range(len(contours_remove), 0, -1):
    del contours[contours_remove[i-1]]
    del histo_data[contours_remove[i-1]]
print(len(contours))
cv2.drawContours(contour_img_2, contours, -1, 255, 3)
cv2.imwrite(f'data/contours/img_contours_2.png', contour_img_2)
'''


# laplacian = cv2.Laplacian(morph_img, cv2.CV_8U)
# sobel = cv2.Sobel(morph_img, cv2.CV_64F, 1, 1, ksize=5)  # try 1,1 instead of 1,0 or 0,1
# sobelx = cv2.Sobel(morph_img, cv2.CV_64F, 1, 0, ksize=5)  # try 1,1 instead of 1,0 or 0,1
# sobely = cv2.Sobel(morph_img, cv2.CV_64F, 0, 1, ksize=5)  # also change kernel size

#images.append(canny)
#titles.append("Canny img")

# blurring
#blur = cv2.blur(canny, (3, 3))
#gaussian_blur = cv2.GaussianBlur(canny, (3, 3), 1)
#median_blur = cv2.medianBlur(canny, 3)
#bilateral_blur = cv2.bilateralFilter(canny, 9, 9 * 2, 9. / 2)

# cv2.imwrite('data/img_canny.png', canny)
# cv2.imwrite('data/img_laplacian.png', laplacian)
# cv2.imwrite('data/img_sobelx.png', sobelx)
# cv2.imwrite('data/img_sobely.png', sobely)

# plot_two_images(input_img, canny, 'Canny')
# plot_two_images(input_img, laplacian, 'Laplacian')
# plot_two_images(input_img, sobelx, 'SobelX')
# plot_two_images(input_img, sobely, 'SobelY')


# Hough Line Transform
'''
img_hough = np.zeros((canny_img.shape[0], canny_img.shape[1]), np.uint8)
lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 200, None, 0, 0)
print(len(lines))

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(img_hough, pt1, pt2, 255, 3, cv2.LINE_AA)
cv2.imwrite(f'data/img_hough.png', img_hough)


# Hough Line P Transform
# for x in range(1, 20):
img_houghp = np.zeros((canny_img.shape[0], canny_img.shape[1]), np.uint8)
linesP = cv2.HoughLinesP(image=canny_img, rho=1, theta=np.pi / 180, threshold=40, minLineLength=10, maxLineGap=10)

print(len(linesP))
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(img_houghp, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)

# cv2.imwrite(f'data/houghp_test/img_houghp_{x}.png', img_houghp)
cv2.imwrite(f'data/img_houghp.png', img_houghp)

'''