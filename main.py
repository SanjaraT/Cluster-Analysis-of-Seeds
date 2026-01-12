import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


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
# df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
# print(df_scaled.head())

kmeans = KMeans(n_clusters = 3, random_state = 42)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

comparison = pd.crosstab(df["Class"],df["cluster"])
print(comparison)

fig, axes = plt.subplots(1,2, figsize =(12,15))
sns.scatterplot(
    data = df,
    x= "Compactness",
    y="Coefficient",
    hue = "Class",
    palette= "Set1",
    s = 70,
    ax = axes[0]
)
axes[0].set_title("Original Seed Classes")
axes[0].set_xlabel("Compactness")
axes[0].set_ylabel("Coefficient")

sns.scatterplot(
    data = df,
    x= "Compactness",
    y="Coefficient",
    hue = "cluster",
    palette= "Set1",
    s = 70,
    ax = axes[1]
)
axes[1].set_title("K-Means Cluster")
axes[1].set_xlabel("Compactness")
axes[1].set_ylabel("Coefficient")

# plt.tight_layout()
# plt.show()

ari = adjusted_rand_score(y_true, clusters)
print("Adjusted Rand Index:", ari)