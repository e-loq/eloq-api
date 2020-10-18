import cv2
import pye57
import pandas as pd
from sklearn.mixture import GaussianMixture
from PIL import Image
import numpy as np
import cv2 as cv
import uuid
from scipy.spatial import ConvexHull
import json_exporter
import copy
from typing import Tuple
import os
cv2 = cv

fname_img_original = f'img_original.png'  # name of the image generated from the pointcloud
minx = None
miny = None
minz = None
maxx = None
maxy = None
maxz = None

def morph(input_img):
    kernel = np.ones((21, 21), np.uint8)
    morph_img = cv2.dilate(input_img, kernel, iterations=1)
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel)
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((20, 20), np.uint8)
    morph_img = cv2.dilate(morph_img, kernel, iterations=1)
    return morph_img


def create_image(e57_path: str):
    resolution = 100
    # TODO: THIS IS JUST FOR TESTING!!!!
    load_file = os.path.exists(f'data/' + fname_img_original)

    print(f'Loading the e57 file...')
    e57 = pye57.E57(e57_path)
    data_raw = e57.read_scan_raw(0)
    print(f'... done.')
    df = pd.DataFrame(data_raw)

    # create a GaussianMixture to get the floor and ceiling cutoff
    print(f'Calculating the cutoff height for the floor and ceiling... ')
    dc = df["cartesianZ"]
    dc_hist = np.histogram(dc.values, bins=80)
    hist_weights = dc_hist[0] / dc_hist[0].sum()
    hist_centers = dc_hist[1]
    sampled_data = np.random.choice(hist_centers[1:], size=10000, p=hist_weights)

    gmm = GaussianMixture(5)
    gmm = gmm.fit(sampled_data.reshape(-1, 1))

    floor_id = np.argmin(gmm.means_)
    ceil_id = np.argmax(gmm.means_)

    floor_cutoff = (gmm.means_[floor_id] + np.sqrt(gmm.covariances_[floor_id])*0.5).reshape(-1)[0]
    ceil_cutoff = (gmm.means_[ceil_id] - np.sqrt(gmm.covariances_[ceil_id])*2.0).reshape(-1)[0]
    print(f'... done.')
    print(f'Floor cutoff: {floor_cutoff}; ceiling cutoff: {ceil_cutoff} \n')

    print(f'Generating the image from the pointcloud...')
    df = df[df.cartesianZ > floor_cutoff]
    df = df[df.cartesianZ < ceil_cutoff-2]

    coords = df[["cartesianX", "cartesianY", "cartesianZ"]].copy()

    minx = coords.cartesianX.min()
    miny = coords.cartesianY.min()
    minz = coords.cartesianZ.min()
    maxx = coords.cartesianX.max()
    maxy = coords.cartesianY.max()
    maxz = coords.cartesianZ.max()

    if not load_file:
        coords.cartesianX -= minx
        coords.cartesianY -= miny
        coords.cartesianZ -= minz
        coords.cartesianX *= resolution
        coords.cartesianX = coords.cartesianX.round().astype(np.int32)
        coords.cartesianY *= resolution
        coords.cartesianY = coords.cartesianY.round().astype(np.int32)
        coords.cartesianZ *= resolution
        coords.cartesianZ = coords.cartesianZ.round().astype(np.int32)

        maxZ = coords.cartesianZ.max()
        minZ = 0

        coords.groupby(["cartesianX", "cartesianY"]).max()
        cartesianX = coords["cartesianX"].values
        cartesianY = coords["cartesianY"].values
        cartesianZ = coords["cartesianZ"].values

        img = np.zeros((cartesianX.max() + 1, cartesianY.max() + 1), np.uint8)

        # normalize between 0 and 255
        for x, y, z in zip(cartesianX, cartesianY, cartesianZ):
            img[x][y] = max(img[x][y], (255 * (z - minZ)) / (maxZ - minZ))
        print(f'... done. /n')
        print(f'Saving the image...')
        img = Image.fromarray(img, "L")
        img.save(f'data/' + fname_img_original)
        print(f'... done.')
        img_input = np.array(img)
        # img_input = cv2.imread(f'data/' + fname_img_original)
    else:
        img_input = cv2.imread(f'data/img_original.png', 0)
    return img_input, {'x': (minx, maxx), 'y': (miny, maxy), 'z': (minz, maxz)}


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


def valentins_part(img_all_colors, img_original, shape_type):
    img_original = morph(img_original)

    all_contours = []
    color_interval_len = 32
    remove_last_interval = 1

    for color_interval_start in range(color_interval_len, 255 - remove_last_interval * color_interval_len,
                                      color_interval_len):
        img = cv.inRange(img_original, color_interval_start, color_interval_start + color_interval_len)

        contours, hierarchy = cv.findContours(img, cv2.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        img_cnt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_outer_cnt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_approx = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_hulls = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_area_filtered = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_cnt_filled = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        def create_graph(vertex, color, image):
            for g in range(0, len(vertex) - 1):
                for y in range(0, len(vertex[0][0]) - 1):
                    cv.circle(image, (vertex[g][0][y], vertex[g][0][y + 1]), 3, (255, 255, 255), -1)
                    cv.line(image, (vertex[g][0][y], vertex[g][0][y + 1]),
                            (vertex[g + 1][0][y], vertex[g + 1][0][y + 1]),
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
                    all_contours.append((currentContour, col_range_end, shape_type))

    return all_contours


def export_json(img_original, all_contours, original_interval_m):
    def map_to_interval(val, old_interval, new_interval):
        A, B = old_interval
        a, b = new_interval
        return (val - A) * (b - a) / (B - A) + a

    interval_x = (0, img_original.shape[1])
    interval_y = (0, img_original.shape[0])
    layers_json_format = []
    for contour, height_intensity, shape_type in all_contours:
        contour_points = []
        for i, point in enumerate(contour):
            contour_points.append({'x': round(map_to_interval(point[0][0], interval_x, original_interval_m['x']), 2),
                                   'y': round(map_to_interval(point[0][1], interval_y, original_interval_m['y']), 2),
                                   'id': i + 1})

        layers_json_format.append(
            {'points': contour_points,
             'height': round(map_to_interval(height_intensity, (0, 255), original_interval_m['z']), 2),
             'shape_type': shape_type,
             'shapeId': str(uuid.uuid1())
             })
    json_exporter.export('data/result.json', layers_json_format, f'data/{fname_img_original}')


def image_processing(e57_path):
    image_path = f'data/img_final.png'
    json_path = f'data/result.json'

    img_input, original_interval_m = create_image(e57_path)
    img_obstacles, img_wall = separate_wall(img_input, 6)

    img_all_colors = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)

    contours_obstacles = valentins_part(img_all_colors, img_obstacles, f'obstacle')
    contours_walls = valentins_part(img_all_colors, img_wall, f'wall')

    cv2.imwrite(image_path, img_all_colors)

    export_json(img_input, contours_obstacles + contours_walls, original_interval_m)

    return image_path, json_path

if __name__ == "__main__":
    image_processing(f'data/CustomerCenter1.e57')
