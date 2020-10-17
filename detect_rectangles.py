import os

import cv2 as cv

cv2 = cv

img = cv.imread('data/img_canny_maxSize.png', 0)
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv2.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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
    cv.line(image, (vertex[len(vertex) - 1][0][0], vertex[len(vertex) - 1][0][1]), (vertex[0][0][0], vertex[0][0][1]),
            color, 2)


#
# for b,cnt in enumerate(contours):
#     if hierarchy[0,b,3] == -1: #<-the mistake might be here
#        approx = cv.approxPolyDP(cnt,0.015*cv.arcLength(cnt,True), True)
#        clr = (255, 0, 0)
#        create_graph(approx, clr) #function for drawing the found contours in the new img
# cv.imwrite('starg.jpg', newimg)


hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

# For each contour, find the bounding rectangle and draw it
for component in zip(contours, hierarchy):
    currentContour = component[0]
    currentHierarchy = component[1]
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
        approx = cv.approxPolyDP(currentContour, 0.0005 * cv.arcLength(currentContour, True), True)
        create_graph(approx, (0, 0, 255), img_approx)
        hull = cv.convexHull(approx)
        create_graph(hull, (0, 0, 255), img_hulls)
        area = cv.contourArea(hull)
        if area >= 10000:
            create_graph(hull, (0, 0, 255), img_area_filtered)

# Finally show the image
if not os.path.exists('data/steps'):
    os.makedirs('data/steps')
cv.imwrite('data/steps/img_1_cnt.png', img_cnt)
cv.imwrite('data/steps/img_2_outer_cnt.png', img_outer_cnt)
cv.imwrite('data/steps/img_3_approx.png', img_approx)
cv.imwrite('data/steps/img_4_hulls.png', img_hulls)
cv.imwrite('data/steps/img_5_area_filtered.png', img_area_filtered)
