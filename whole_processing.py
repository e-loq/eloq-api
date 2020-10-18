import numpy as np
import matplotlib.pyplot as plt
import cv2
import pye57
import pandas as pd
from sklearn.mixture import GaussianMixture
from PIL import Image

# some starting arguments
load_from_e57 = True
resolution = 100
fname_img_input = f'img_{resolution}_input.png'
fname_img_morph = f'img_{resolution}_morph.png'
fname_img_canny = f'img_{resolution}_canny.png'

img_input = None
img_morph = None
img_canny = None
if load_from_e57:
    print(f'Loading the e57 file...')
    e57 = pye57.E57('data/CustomerCenter1.e57')
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
    gmm = gmm.fit(sampled_data.reshape(-1,1))

    floor_id = np.argmin(gmm.means_)
    ceil_id = np.argmax(gmm.means_)

    floor_cutoff = (gmm.means_[floor_id] + np.sqrt(gmm.covariances_[floor_id])*0.5).reshape(-1)[0]
    ceil_cutoff = (gmm.means_[ceil_id] - np.sqrt(gmm.covariances_[ceil_id])*2.0).reshape(-1)[0]
    print(f'... done.')
    print(f'Floor cutoff: {floor_cutoff}; ceiling cutoff: {ceil_cutoff} /n')

    print(f'Generating the image from the pointcloud...')
    df = df[df.cartesianZ > floor_cutoff]
    df = df[df.cartesianZ < ceil_cutoff-2]

    coords = df[["cartesianX", "cartesianY", "cartesianZ"]].copy()

    minx = coords.cartesianX.min()
    miny = coords.cartesianY.min()
    minz = coords.cartesianZ.min()
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
    img.save(f'data/' + fname_img_input)
    print(f'... done.')
    img_input = np.array(img)
else:
    img_input = cv2.imread(f'data/' + fname_img_input)

print(f'Image is ready. Starting the processing steps.')
print(f'Applying morphological operations...')
kernel = np.ones((7, 7), np.uint8)
img_morph = cv2.dilate(img_input, kernel, iterations=1)
img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)
img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)
print(f'.. done.')
cv2.imwrite(f'data/' + fname_img_morph, img_morph)

print(f'Applying the Canny edge detector...')
img_canny = cv2.Canny(img_morph, 75, 250, apertureSize=3, L2gradient=True)
cv2.imwrite(f'data/' + fname_img_canny, img_canny)
print(f'... done.')

# spacing = np.zeros((), np.uint8)
# img_final = np.vstack((img_input, spacing, img_morph, spacing, img_canny))
# cv2.imwrite(f'data/img_all_combined', img_final)
exit(0)
