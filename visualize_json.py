import argparse
import json
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--json', required=True)
parser.add_argument('--background-img', required=True)
args = parser.parse_args()


def create_graph(vertex, color, image):
    for g in range(0, len(vertex) - 1):
        for y in range(0, len(vertex[0][0]) - 1):
            cv.circle(image, (vertex[g][0][y], vertex[g][0][y + 1]), 3, (255, 255, 255), -1)
            cv.line(image, (vertex[g][0][y], vertex[g][0][y + 1]), (vertex[g + 1][0][y], vertex[g + 1][0][y + 1]),
                    color, 2)
    cv.line(image, (vertex[len(vertex) - 1][0][0], vertex[len(vertex) - 1][0][1]),
            (vertex[0][0][0], vertex[0][0][1]),
            color, 2)


def map_to_interval(val, old_interval, new_interval):
    A, B = old_interval
    a, b = new_interval
    return (val - A) * (b - a) / (B - A) + a


with open(args.json) as f:
    j = json.load(f)

all_shapes = []
shape_heights = []
all_x, all_y = [], []
for shape in j['layers']:
    shape_points = []
    for point in shape['points']:
        shape_points.append((point['x'], point['y']))
        all_x.append(point['x'])
        all_y.append(point['y'])
    all_shapes.append(shape_points)
    shape_heights.append(shape['height'])

background_img = cv.cvtColor(cv.imread(args.background_img, 0), cv.COLOR_GRAY2RGB)

intervals_meters = {'x': (min(all_x), max(all_x)),
                    'y': (min(all_y), max(all_y)),
                    'z': (min(shape_heights), max(shape_heights))}

interval_img = {'x': (0, background_img.shape[1]),
                'y': (0, background_img.shape[0]),
                'z': (0, 255)}

for shape, height in zip(all_shapes, shape_heights):
    shape_new_interval = []
    for point in shape:
        shape_new_interval.append([[int(map_to_interval(point[0], intervals_meters['x'], interval_img['x'])),
                                    int(map_to_interval(point[1], intervals_meters['y'], interval_img['y']))]])
    scaled_height = int(map_to_interval(height, intervals_meters['z'], interval_img['z']))
    shape_color = (0, 255 - scaled_height, scaled_height)
    cv.fillPoly(background_img, pts=[np.array(shape_new_interval)], color=shape_color)

cv.imwrite('data/result_check.png', background_img)
