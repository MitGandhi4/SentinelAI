import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv("data/processed/cleaned.csv")

# ── Plot 1: Attack type distribution ──────────────────────────────
label_counts = df["Label"].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=label_counts.values, y=label_counts.index, palette="Reds_r")
plt.title("Attack Type Distribution in CICIDS-2017")
plt.xlabel("Number of Rows")
plt.ylabel("Label")
plt.tight_layout()
plt.savefig("notebooks/plot1_label_distribution.png")
plt.show()
print("Plot 1 saved.")

# ── Plot 2: Feature distributions — benign vs attack ──────────────
features_to_plot = ["Flow Duration", "Total Fwd Packets",
                    "Total Backward Packets", "Flow Bytes/s"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    benign_data = df[df["is_attack"] == 0][feature]
    attack_data = df[df["is_attack"] == 1][feature]

    axes[i].hist(benign_data, bins=50, alpha=0.5, label="Benign", color="blue")
    axes[i].hist(attack_data, bins=50, alpha=0.5, label="Attack", color="red")
    axes[i].set_title(feature)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    axes[i].legend()

# ── Plot 3: Correlation heatmap ────────────────────────────────────
# Sample 10,000 rows for speed — computing correlations on 2.8M rows is slow
sample_df = df.sample(10000, random_state=42)

# Only use numeric columns
numeric_df = sample_df.select_dtypes(include="number")

# Drop the binary label we created since it would dominate the heatmap
numeric_df = numeric_df.drop(columns=["is_attack"])

plt.figure(figsize=(20, 16))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", center=0,
            linewidths=0, xticklabels=False, yticklabels=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("notebooks/plot3_correlation_heatmap.png")
plt.show()
print("Plot 3 saved.")

# ── Plot 4: Constant and near-constant feature check ──────────────
# A feature with zero variance is useless — it's the same value for every row
numeric_df2 = df.select_dtypes(include="number").drop(columns=["is_attack"])

variances = numeric_df2.var()
low_variance = variances[variances < 0.01].index.tolist()

print(f"\nFeatures with near-zero variance (useless for modeling): {low_variance}")

# Plot variance of all features
plt.figure(figsize=(14, 6))
variances.sort_values().plot(kind="bar", color="steelblue")
plt.title("Feature Variances (low = potentially useless)")
plt.ylabel("Variance")
plt.xticks([])
plt.tight_layout()
plt.savefig("notebooks/plot4_feature_variances.png")
plt.show()
print("Plot 4 saved.")

plt.suptitle("Feature Distributions: Benign vs Attack", fontsize=14)
plt.tight_layout()
plt.savefig("notebooks/plot2_feature_distributions.png")
plt.show()
print("Plot 2 saved.")