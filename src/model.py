import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load cleaned dataset
df = pd.read_csv("./data/final/merged_cleaned_dataset.csv")

# Step 2: Set index (if not already)
df.set_index("Area", inplace=True)

# Step 3: Handle missing values (if any)
df.fillna(0, inplace=True)

# Step 4: Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Step 5: PCA to reduce to a single Development Index
pca = PCA(n_components=1)
dev_index = pca.fit_transform(scaled_features)
df["Development_Index"] = dev_index

# Step 6: KMeans clustering (let's try 3 groups)
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Step 7: Save to CSV
df.to_csv("clustered_development_index.csv")

# Step 8: Visualization workaround (hue not supported in histplot for wide-form with numeric hue)
plt.figure(figsize=(10, 6))
for cluster in sorted(df['Cluster'].unique()):
    sns.histplot(df[df['Cluster'] == cluster]['Development_Index'], bins=30, kde=True, label=f"Cluster {cluster}", element="step")
plt.title("Distribution of Development Index by Cluster")
plt.xlabel("Development Index")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Show cluster means
print(df.groupby("Cluster").mean(numeric_only=True).round(2))

plt.figure(figsize=(10, 6))
for cluster in sorted(df['Cluster'].unique()):
    sns.histplot(
        data=df[df['Cluster'] == cluster],
        x='Development_Index',
        bins=30,
        kde=True,
        label=f"Cluster {cluster}",
        alpha=0.5  # transparency
    )

plt.title("Distribution of Development Index by Cluster")
plt.xlabel("Development Index")
plt.ylabel("Count")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# Optional: 2D PCA visualization
pca_2d = PCA(n_components=2)
pca_components = pca_2d.fit_transform(scaled_features)

df["PC1"] = pca_components[:, 0]
df["PC2"] = pca_components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="tab10")
plt.title("2D PCA Projection with Cluster Labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
