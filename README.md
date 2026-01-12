📌 Overview

This project explores the UCI Seeds dataset, which contains geometric measurements of wheat kernels from three different varieties. The goal was to analyze the data structure and apply unsupervised learning to discover natural groupings within the dataset.

📊 Dataset

Source: UCI Machine Learning Repository
Samples: 210
Features: 7 numerical attributes
Classes: 3 wheat seed varieties

🧠 Methodology

Performed basic exploratory data analysis (EDA)
Standardized features using StandardScaler
Applied K-Means clustering with k = 3
Compared clusters with original labels using:
Contingency table
Scatter plot visualization
Adjusted Rand Index (ARI)


📏 Evaluation Metric

Adjusted Rand Index (ARI): 0.79
This score indicates a strong agreement between the K-Means clusters and the true seed classes, showing that the dataset has clear natural separability.