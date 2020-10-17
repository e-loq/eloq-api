import pye57
import numpy as np
import pandas as pd

data_raw = pye57.E57("data/CustomerCenter1 1.e57")
data = data_raw.read_scan_raw(0)

collist = []
colnames= []

for key, val in data.items():
    collist.append(val.reshape([-1, 1]))
    colnames.append(key)

df = np.hstack(collist)
df = pd.DataFrame(df, columns=colnames)

df["intensity"] = df.intensity.astype(np.float)
df.colorBlue = df.colorBlue.astype(np.int16)
df.colorRed = df.colorRed.astype(np.int16)
df.colorGreen = df.colorGreen.astype(np.int16)
df.cartesianX = df.cartesianX.astype(np.float32)
df.cartesianY = df.cartesianY.astype(np.float32)
df.cartesianZ = df.cartesianZ.astype(np.float32)
df.to_pickle("data/raw_data.pkl")
# df.head()
# df.cartesianX.describe()
# df.cartesianY.describe()
# df.cartesianZ.describe()
df_reduced = df[df.cartesianY <= -2.896500]
df_reduced.to_pickle("data/df_reduced.pkl")

del df, data_raw, data