import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from PIL import Image


def get_distributions(df:pd.DataFrame, col:str):
    dc = df[col]
    dc.plot.hist(bins=90)
    # plt.show()
    # plt.close()
    return dc


def generate_image(df:pd.DataFrame):
    coords = df[["cartesianX", "cartesianY"]].copy()
    minx = coords.cartesianX.min()
    miny = coords.cartesianY.min()

    coords.cartesianX -= minx
    coords.cartesianY -= miny

    coords.cartesianX *= 1000
    coords.cartesianX = coords.cartesianX.round().astype(np.int32)

    coords.cartesianY *= 1000
    coords.cartesianY = coords.cartesianY.round().astype(np.int32)

    maxx = coords.cartesianX.max()
    maxy = coords.cartesianY.max()

    matrix_bw = np.zeros([maxx + 1, maxy + 1], np.uint8)
    matrix_bw[coords.cartesianX.values, coords.cartesianY.values] = 255
    img = Image.fromarray(matrix_bw, "L")
    return img

df = pd.read_pickle("data/df_reduced.pkl")
dc = get_distributions(df, "cartesianZ")


dc_hist = np.histogram(dc.values, bins=80)
hist_weights = dc_hist[0] / dc_hist[0].sum()
hist_centers = dc_hist[1]
sampled_data = np.random.choice(hist_centers[1:], size=10000, p=hist_weights)

gmm = GaussianMixture(5)
gmm = gmm.fit(sampled_data.reshape(-1,1))

floor_id = np.argmin(gmm.means_)
ceil_id = np.argmax(gmm.means_)

floor_cutoff = (gmm.means_[floor_id] + np.sqrt(gmm.covariances_[floor_id])*2).reshape(-1)[0]
ceil_cutoff = (gmm.means_[ceil_id] - np.sqrt(gmm.covariances_[ceil_id])*2).reshape(-1)[0]

df = df[df.cartesianZ > floor_cutoff]
df = df[df.cartesianZ < ceil_cutoff-2]
df_plot = df[["cartesianX", "cartesianY", "intensity"]]
df_plot.plot.scatter("cartesianX", "cartesianY", s=1, figsize=(15, 10))
# plt.show()

img = generate_image(df)
# (width, height) = (img.width // 4, img.height // 4)
# img = img.resize((width, height))
img.save('data/img_original.png')