import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

cols = ["Area","Perimeter","Compactness","Length","Width","Coefficient","Groove","Class"]
df = pd.read_csv("seeds_dataset.txt",names = cols, sep = "\s+")
print(df.head())