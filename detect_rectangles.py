import os
import numpy as np
import cv2 as cv
import uuid
from scipy.spatial import ConvexHull
import json_exporter
import copy
from typing import Tuple
cv2 = cv

# TODO don't hardcode
original_interval = {'x': (0, 16), 'y': (0, 9), 'z': (0, 4)}


def morph(input_img):
    kernel = np.ones((21, 21), np.uint8)
    morph_img = cv2.dilate(input_img, kernel, iterations=1)
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel)
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((20, 20), np.uint8)
    morph_img = cv2.dilate(morph_img, kernel, iterations=1)
    return morph_img


# Test commit

def separate_wall(img_:np.ndarray, threshold:np.float=3.0)-> Tuple[np.ndarray, np.ndarray]:
    """
    Naive wall vs obstacles separations
    Parameters
    ----------
    img_ : the image input matrix
    threshold : percentage threshold to define wall vs everytihng else

    Returns : Tuple[obstacles:np.ndarray, walls: np.ndarray]
    -------
    """

    img_original = copy.deepcopy(img_)
    img_wall = copy.deepcopy(img_) * 0
    img_obstacles = copy.deepcopy(img_)

    def point_to_line(point, p1, p2):
        # line between p1, p2

        # check calculate shapes
        assert p1.shape[0] == 2
        assert p2.shape[0] == 2
        assert point.shape[1] == 2

        zaeler = np.abs((p2[1] - p1[1]) * point[:, 0] - (p2[0] - p1[0]) * point[:, 1] + p2[0] * p1[1] - p2[1] * p1[0])
        nenner = np.sqrt(np.square(p2[1] - p1[1]) + np.square(p2[0] - p1[0]))
        return zaeler / nenner

    point_coords_original_img = np.where(img_original > 3)
    point_coords_original_img = np.hstack([point_coords_original_img[0].reshape([-1, 1]),
                                           point_coords_original_img[1].reshape([-1, 1])])
    hull = ConvexHull(point_coords_original_img)
    hull_points = point_coords_original_img[hull.vertices]

    for ix in range(hull_points.shape[0]):
        start = hull_points[ix, :]
        if (ix + 1) == hull_points.shape[0]:
            end = hull_points[0, :]
        else:
            end = hull_points[ix + 1, :]
        distances = point_to_line(point_coords_original_img, start, end)
        thresh = np.percentile(distances, threshold)  # 0.001 - one promile
        zero_point_coords = point_coords_original_img[np.where(distances < thresh)]
        img_obstacles[zero_point_coords[:, 0], zero_point_coords[:, 1]] = 0  # inverted?
        img_wall[zero_point_coords[:, 0], zero_point_coords[:, 1]] = img_original[zero_point_coords[:, 0], zero_point_coords[:, 1]]

    return img_obstacles, img_wall

# height map
background_img_path = 'data/entire_hall.png'
img_original_ = cv.imread(background_img_path, 0)


img_obstacles, img_wall = separate_wall(img_original_, 6)

# TODO set this to run with the correct input depending on what you would like to use
img_original = img_obstacles
img_original = img_wall


img_all_colors = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)

img_original = morph(img_original)

# TODO find color thresholds using kmeans

all_contours = []

color_interval_len = 32
remove_last_interval = 1
for color_interval_start in range(color_interval_len, 255 - remove_last_interval * color_interval_len,
                                  color_interval_len):
    img = cv.inRange(img_original, color_interval_start, color_interval_start + color_interval_len)

    contours, hierarchy = cv.findContours(img, cv2.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(img, contours, -1, (0,255,0), 3)
    #
    # cv.imwrite('data/result.png', img)

    img_cnt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_outer_cnt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_approx = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_hulls = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_area_filtered = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_cnt_filled = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


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

        cv.fillPoly(img_cnt_filled, pts=[currentContour], color=(0, 255, 0))

        if currentHierarchy[3] < 0:
            create_graph(currentContour, (0, 0, 255), img_outer_cnt)
            approx = cv.approxPolyDP(currentContour, 0.001 * cv.arcLength(currentContour, True), True)
            create_graph(approx, (0, 0, 255), img_approx)
            hull = approx
            create_graph(hull, (0, 0, 255), img_hulls)
            area = cv.contourArea(approx)
            if area >= 10000:
                create_graph(hull, (0, 0, 255), img_area_filtered)
                create_graph(hull, (0, 0, 255), img_all_colors)
                col_range_end = color_interval_start + color_interval_len
                all_contours.append((currentContour, col_range_end))

    # Finally show the image
    if not os.path.exists('data/steps'):
        os.makedirs('data/steps')
    cv.imwrite(f'data/steps/img_r{color_interval_start}_1_cnt.png', img_cnt)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_2_outer_cnt.png', img_outer_cnt)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_3_approx.png', img_approx)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_4_hulls.png', img_hulls)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_5_area_filtered.png', img_area_filtered)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_5_img_all_colors.png', img_all_colors)
    cv.imwrite(f'data/steps/img_r{color_interval_start}_5_img_cnt_filled.png', img_cnt_filled)

cv.imwrite(f'data/steps/img_final_all_contours.png', img_all_colors)


def map_to_interval(val, old_interval, new_interval):
    A, B = old_interval
    a, b = new_interval
    return (val - A) * (b - a) / (B - A) + a


interval_x = (0, img_original.shape[1])
interval_y = (0, img_original.shape[0])
layers_json_format = []
for contour, height_intensity in all_contours:
    contour_points = []
    for i, point in enumerate(contour):
        contour_points.append({'x': round(map_to_interval(point[0][0], interval_x, original_interval['x']), 2),
                               'y': round(map_to_interval(point[0][1], interval_y, original_interval['y']), 2), 'id': i + 1})

    layers_json_format.append(
        {'points': contour_points,
         'height': round(map_to_interval(height_intensity, (0, 255), original_interval['z']), 2),
         'shape_type': 'obstacle',
         'shapeId': str(uuid.uuid1())
         })

json_exporter.export('data/result.json', layers_json_format, background_img_path)
