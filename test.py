import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from PIL import Image
import pye57


def get_distributions(df:pd.DataFrame, col:str):
    dc = df[col]
    return dc


e57 = pye57.E57('data/CustomerCenter1.e57')
data_raw = e57.read_scan_raw(0)
df = pd.DataFrame(data_raw)

# df = pd.read_pickle("data/df_reduced.pkl")
dc = get_distributions(df, "cartesianZ")

dc_hist = np.histogram(dc.values, bins=80)
hist_weights = dc_hist[0] / dc_hist[0].sum()
hist_centers = dc_hist[1]
sampled_data = np.random.choice(hist_centers[1:], size=10000, p=hist_weights)

gmm = GaussianMixture(5)
gmm = gmm.fit(sampled_data.reshape(-1,1))

floor_id = np.argmin(gmm.means_)
ceil_id = np.argmax(gmm.means_)

floor_cutoff = (gmm.means_[floor_id] + np.sqrt(gmm.covariances_[floor_id])*0.0).reshape(-1)[0]
ceil_cutoff = (gmm.means_[ceil_id] - np.sqrt(gmm.covariances_[ceil_id])*2.0).reshape(-1)[0]

df = df[df.cartesianZ > floor_cutoff]
df = df[df.cartesianZ < ceil_cutoff-2]

coords = df[["cartesianX", "cartesianY", "cartesianZ"]].copy()

minx = coords.cartesianX.min()
miny = coords.cartesianY.min()
minz = coords.cartesianZ.min()

coords.cartesianX -= minx
coords.cartesianY -= miny
coords.cartesianZ -= minz

amount = 100
coords.cartesianX *= amount
coords.cartesianX = coords.cartesianX.round().astype(np.int32)

coords.cartesianY *= amount
coords.cartesianY = coords.cartesianY.round().astype(np.int32)

coords.cartesianZ *= amount
coords.cartesianZ = coords.cartesianZ.round().astype(np.int32)

maxZ = coords.cartesianZ.max()
minZ = 0
coords.groupby(["cartesianX", "cartesianY"]).max()
cartesianX = coords["cartesianX"].values
cartesianY = coords["cartesianY"].values
cartesianZ = coords["cartesianZ"].values

img = np.zeros((cartesianX.max() + 1, cartesianY.max() + 1), np.uint8)

for x, y, z in zip(cartesianX, cartesianY, cartesianZ):
    img[x][y] = max(img[x][y], (255 * (z - minZ)) / (maxZ - minZ))

img = Image.fromarray(img, "L")
img.save('data/image_100_floorincluded.png')
