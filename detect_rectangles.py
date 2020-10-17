import os
import numpy as np
import cv2 as cv

cv2 = cv


def morph(morph_img):
    kernel = np.ones((3, 3), np.uint8)
    morph_img = cv2.dilate(morph_img, kernel, iterations=3)
    morph_img = cv2.erode(morph_img, kernel, iterations=1)
    return morph_img

img_original = cv.imread('data/height_map.png', 0)

# TODO find color thresholds using kmeans

color_interval_len = 32
for color_interval_start in range(color_interval_len, 255, color_interval_len):
    img = cv.inRange(img_original, color_interval_start, color_interval_start + color_interval_len)

    img = morph(img)
    contours, hierarchy = cv.findContours(img, cv2.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(img, contours, -1, (0,255,0), 3)
    #
    # cv.imwrite('data/result.png', img)

    img_cnt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_outer_cnt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_approx = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_hulls = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_area_filtered = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    #
    def create_graph(vertex, color, image):
        for g in range(0, len(vertex) - 1):
            for y in range(0, len(vertex[0][0]) - 1):
                cv.circle(image, (vertex[g][0][y], vertex[g][0][y + 1]), 3, (255, 255, 255), -1)
                cv.line(image, (vertex[g][0][y], vertex[g][0][y + 1]), (vertex[g + 1][0][y], vertex[g + 1][0][y + 1]),
                        color, 2)
        cv.line(image, (vertex[len(vertex) - 1][0][0], vertex[len(vertex) - 1][0][1]),
                (vertex[0][0][0], vertex[0][0][1]),
                color, 2)


    #
    # for b,cnt in enumerate(contours):
    #     if hierarchy[0,b,3] == -1: #<-the mistake might be here
    #        approx = cv.approxPolyDP(cnt,0.015*cv.arcLength(cnt,True), True)
    #        clr = (255, 0, 0)
    #        create_graph(approx, clr) #function for drawing the found contours in the new img
    # cv.imwrite('starg.jpg', newimg)

    hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions
    if not isinstance(hierarchy, np.ndarray) or len(hierarchy.shape) == 1:
        continue

    # For each contour, find the bounding rectangle and draw it
    for currentContour, currentHierarchy in zip(contours, hierarchy):
        # x, y, w, h = cv2.boundingRect(currentContour)
        # if currentHierarchy[2] < 0:
        #     # these are the innermost child components
        #     # cv2.rectangle(newimg, (x, y), (x + w, y + h), (0, 0, 255), 20)
        #     pass
        # elif currentHierarchy[3] < 0:
        #     # these are the outermost parent components
        #     cv2.rectangle(newimg, (x, y), (x + w, y + h), (0, 255, 0), 20)

        create_graph(currentContour, (0, 0, 255), img_cnt)

        if currentHierarchy[3] < 0:
            create_graph(currentContour, (0, 0, 255), img_outer_cnt)
            approx = cv.approxPolyDP(currentContour, 0.01 * cv.arcLength(currentContour, True), True)
            create_graph(approx, (0, 0, 255), img_approx)
            hull = approx
            create_graph(hull, (0, 0, 255), img_hulls)
            area = cv.contourArea(approx)
            if area >= 10000:
                create_graph(hull, (0, 0, 255), img_area_filtered)

    # Finally show the image
    if not os.path.exists('data/steps'):
        os.makedirs('data/steps')
    cv.imwrite(f'data/steps/img_r{color_interval_start}_1_cnt.png', img_cnt)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_2_outer_cnt.png', img_outer_cnt)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_3_approx.png', img_approx)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_4_hulls.png', img_hulls)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_5_area_filtered.png', img_area_filtered)
