import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler


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
# plt.show()

X = df.drop("Class", axis = 1)
y_true = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(df_scaled.head())