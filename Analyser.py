import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import periodogram, detrend
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
import warnings
warnings.filterwarnings("ignore")

start = time.time()

INPUT_FILE = "pulmobench_combined.csv"
OUTPUT_DIR = "Output"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df.columns = [c.lower().replace(" ", "_") for c in df.columns]
df["escalation_required"] = df["escalation_required"].astype(int)

target = "risk_score"
categorical = ["sex", "risk_tier", "urgency", "perturbation_type"]
numeric = ["age", "escalation_required"]

df_model = df[numeric + [target] + categorical]
df_model = pd.get_dummies(df_model, columns=categorical, drop_first=True)
feature_cols = [c for c in df_model.columns if c != target]

scaler = StandardScaler()
scaled = scaler.fit_transform(df_model[feature_cols])

pca = PCA(n_components=min(6, scaled.shape[1]))
pca_data = pca.fit_transform(scaled)

def savefig(fig, name):
    fig.savefig(os.path.join(CHART_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---- Generate Plots ----

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.set_title("Cumulative PCA Explained Variance")
ax.set_xlabel("Principal Components")
ax.set_ylabel("Cumulative Explained Variance Ratio")
savefig(fig, "pca_variance.png")

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(pca_data[:,0], pca_data[:,1], s=15)
ax.set_title("2D PCA Projection")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
savefig(fig, "pca_projection.png")

iso = IsolationForest(contamination=0.05, random_state=42)
anom = iso.fit_predict(scaled)
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(range(len(df)), df[target], c=anom, cmap="coolwarm", s=15)
ax.set_title("Anomaly Detection (Isolation Forest)")
ax.set_xlabel("Case Index")
ax.set_ylabel("Risk Score")
savefig(fig, "anomaly.png")

kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
clusters = kmeans.fit_predict(scaled)
sil = silhouette_score(scaled, clusters)
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(pca_data[:,0], pca_data[:,1], c=clusters, cmap="viridis", s=15)
ax.set_title("KMeans Cluster Segmentation")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
savefig(fig, "cluster.png")

series = df[target].values
freq, power = periodogram(detrend(series))
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(freq[1:], power[1:])
ax.set_yscale("log")
ax.set_title("Fourier Frequency Spectrum")
ax.set_xlabel("Frequency")
ax.set_ylabel("Log Power")
savefig(fig, "fourier.png")

rolling_mean = df[target].rolling(max(3, len(df)//20)).mean()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(df[target], alpha=0.4)
ax.plot(rolling_mean)
ax.set_title("Rolling Mean of Risk Score")
ax.set_xlabel("Index")
ax.set_ylabel("Risk Score")
savefig(fig, "rolling_mean.png")

rolling_std = df[target].rolling(max(3, len(df)//20)).std()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(rolling_std)
ax.set_title("Rolling Volatility of Risk Score")
ax.set_xlabel("Index")
ax.set_ylabel("Standard Deviation")
savefig(fig, "rolling_volatility.png")

returns = df[target].pct_change().fillna(0)
cum = (1 + returns).cumprod()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cum)
ax.set_title("Cumulative Growth of Risk Score")
ax.set_xlabel("Index")
ax.set_ylabel("Growth Factor")
savefig(fig, "cumulative_growth.png")

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df_model.corr(), cmap="viridis", ax=ax)
ax.set_title("Feature Correlation Heatmap")
savefig(fig, "correlation.png")

fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(df[target], bins=30, ax=ax)
ax.set_title("Risk Score Distribution")
ax.set_xlabel("Risk Score")
ax.set_ylabel("Frequency")
savefig(fig, "distribution.png")

mi = mutual_info_regression(df_model[feature_cols], df[target])
imp = pd.Series(mi, index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(10,6))
imp.plot(kind="barh", ax=ax)
ax.set_title("Feature Importance (Mutual Information)")
ax.set_xlabel("Mutual Information Score")
savefig(fig, "feature_importance.png")

# ---- PDF REPORT ----

styles = getSampleStyleSheet()
body = ParagraphStyle("Body", parent=styles["BodyText"], leading=16)

doc = SimpleDocTemplate(os.path.join(OUTPUT_DIR, "PulmoBench_Full_Report.pdf"))
elements = []

elements.append(Paragraph("PulmoBench Comprehensive Analytical Intelligence Report", styles["Heading1"]))
elements.append(Spacer(1, 12))

elements.append(Paragraph(f"""
This report analyzes {len(df)} patient records using dimensional reduction,
clustering, anomaly detection, spectral analysis, and statistical profiling.
The silhouette score of {round(sil,3)} indicates moderate structural separation
between identified clusters.
""", body))

elements.append(PageBreak())

explanations = {
"pca_variance": "Explains how variance accumulates across principal components. Early saturation indicates strong latent dimensional structure.",
"pca_projection": "Visualizes structural grouping in reduced dimensional space. Cluster separation indicates population heterogeneity.",
"anomaly": "Highlights statistically rare cases based on multivariate isolation depth.",
"cluster": f"Segments population into 4 archetypes. Silhouette score {round(sil,3)} measures cluster cohesion.",
"fourier": "Reveals periodic structure embedded in risk score ordering.",
"rolling_mean": "Smooths short-term fluctuations to expose longer-term trend patterns.",
"rolling_volatility": "Measures temporal instability in risk behavior.",
"cumulative_growth": "Shows compounded proportional change in risk trajectory.",
"correlation": "Quantifies linear interdependencies among predictors.",
"distribution": "Displays central tendency and skewness of risk score.",
"feature_importance": "Ranks nonlinear predictive influence of features on risk score."
}

for chart in sorted(os.listdir(CHART_DIR)):
    name = chart.replace(".png","")
    elements.append(Paragraph(name.replace("_"," ").title(), styles["Heading2"]))
    elements.append(Spacer(1,10))
    img = Image(os.path.join(CHART_DIR, chart))
    img._restrictSize(6*inch, 4*inch)
    elements.append(img)
    elements.append(Spacer(1,12))
    elements.append(Paragraph(explanations.get(name,"Analysis unavailable."), body))
    elements.append(PageBreak())

doc.build(elements)

print("Full analytical PDF with explanations generated.")
print("Execution Time:", round(time.time()-start,2))