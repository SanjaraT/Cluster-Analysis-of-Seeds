import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

cols = ["Area","Perimeter","Compactness","Length","Width","Coefficient","Groove","Class"]
df = pd.read_csv("seeds_dataset.txt",names = cols, sep = "\s+")
# print(df.head())

df["Class"].value_counts().plot(kind="bar")
plt.xlabel("Seed Class")
plt.ylabel("Counts")
plt.title("Class Distribution")
# plt.show()

sns.pairplot(df, hue ="Class",diag_kind="hist")
# plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.drop("Class",axis = 1).corr(),
            annot= True,
            cmap= "coolwarm",
            fmt = ".2f")
plt.title("Correlation Matrix")
plt.show()
