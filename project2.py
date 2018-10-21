import numpy as np
import pandas as pd

img_features = pd.read_csv("./HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv", index_col="img_id")

img_features = img_features[:,1:]

print(img_features)