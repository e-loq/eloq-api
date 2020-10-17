import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


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


input_img = cv2.imread(f'data/image.png')

images = []
images.append(input_img)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(input_img, kernel, iterations=1)
dilation = cv2.dilate(input_img, kernel, iterations=1)
opening = cv2.morphologyEx(input_img, cv2.MORPH_OPEN, kernel)  # erosion followed by dilation
closing = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, kernel)  # dilation follows by erosion



erosion = cv2.erode(input_img, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)
# erosion = cv2.erode(dilation, kernel, iterations=1)
# dilation = cv2.dilate(erosion, kernel, iterations=1)
# cv2.imwrite(f'data/img_closing.png', dilation)
images.append(dilation)
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
canny = cv2.Canny(dilation, 200, 400, apertureSize=3, L2gradient=False)
laplacian = cv2.Laplacian(closing, cv2.CV_8U)
sobel = cv2.Sobel(closing, cv2.CV_64F, 1, 1, ksize=5)  # try 1,1 instead of 1,0 or 0,1
sobelx = cv2.Sobel(closing, cv2.CV_64F, 1, 0, ksize=5)  # try 1,1 instead of 1,0 or 0,1
sobely = cv2.Sobel(closing, cv2.CV_64F, 0, 1, ksize=5)  # also change kernel size


images.append(canny)

# blurring
blur = cv2.blur(canny, (3, 3))
gaussian_blur = cv2.GaussianBlur(canny, (3, 3), 1)
median_blur = cv2.medianBlur(canny, 3)
bilateral_blur = cv2.bilateralFilter(canny, 9, 9 * 2, 9. / 2)

# cv2.imwrite('data/img_canny.png', canny)
# cv2.imwrite('data/img_laplacian.png', laplacian)
# cv2.imwrite('data/img_sobelx.png', sobelx)
# cv2.imwrite('data/img_sobely.png', sobely)

# plot_two_images(input_img, canny, 'Canny')
# plot_two_images(input_img, laplacian, 'Laplacian')
# plot_two_images(input_img, sobelx, 'SobelX')
# plot_two_images(input_img, sobely, 'SobelY')

'''
lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

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
        cv2.line(canny, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

show_image(canny)
'''

linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(canny, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)


# show_images(images)