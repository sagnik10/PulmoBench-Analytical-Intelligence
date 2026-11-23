<!-- SAGNIK10-STYLE-HEADER -->
<div align="center">
  <a href="https://github.com/sagnik10"><img src="https://img.shields.io/badge/Sagnik_Sen-Data%20Analytics%20%26%20ML-275cce?style=flat-square" alt="Category" /></a>
  <img src="https://img.shields.io/badge/Stack-Python-192a43?style=flat-square" alt="Stack" />
  <img src="https://img.shields.io/badge/License-MIT-54647a?style=flat-square" alt="License" />
</div>

<p align="center">
  <a href="https://sagnik10.github.io/SagnikSenPortfolio/">Portfolio</a> ·
  <a href="https://github.com/sagnik10">GitHub Profile</a> ·
  <a href="https://github.com/sagnik10/sagnik-sen-antigravity-plugin">Antigravity Plugin</a>
</p>

---
<!-- /SAGNIK10-STYLE-HEADER -->
# PulmoBench Analytical Intelligence

A comprehensive analytical intelligence pipeline for the PulmoBench dataset that performs dimensional reduction, anomaly detection, clustering, statistical profiling, and automated PDF report generation.

This system transforms structured clinical risk data into interpretable analytical insights using machine learning, statistical modeling, and automated reporting.

---

## Overview

PulmoBench Analytical Intelligence is an end-to-end automated analytics engine designed to extract structural patterns from patient-level risk data.

The pipeline includes:

- Principal Component Analysis (PCA)
- Isolation Forest Anomaly Detection
- KMeans Risk Segmentation
- Fourier Spectral Analysis
- Rolling Trend & Volatility Analysis
- Correlation Heatmap
- Mutual Information Feature Importance
- Automated Multi-Page PDF Intelligence Report

---

## Repository Structure

PulmoBench-Analytical-Intelligence/
│
├── Data_Converted.py  
├── pulmobench_combined.csv  
├── Output/  
│   ├── charts/  
│   └── PulmoBench_Full_Report.pdf  
└── README.md  

---

## Installation

Install dependencies:

pip install numpy pandas matplotlib seaborn scipy scikit-learn reportlab

---

## How to Run

1. Place `pulmobench_combined.csv` in the same directory as the script.
2. Run:

python Data_Converted.py

After execution:

- An `Output/` folder will be created
- All analytical charts will be generated
- A structured PDF analytical report will be produced

---

## Analytical Modules

### 1. Principal Component Analysis (PCA)
Identifies latent structural dimensions and variance concentration.

### 2. Anomaly Detection
Isolation Forest detects statistically rare or extreme patient configurations.

### 3. Risk Segmentation
KMeans clustering stratifies the dataset into structural archetypes.

### 4. Fourier Spectral Analysis
Detects periodic or oscillatory patterns within risk score distributions.

### 5. Rolling Mean & Volatility
Evaluates smoothed trends and temporal instability in risk behavior.

### 6. Correlation Heatmap
Quantifies linear dependencies between predictive features.

### 7. Feature Importance
Mutual Information ranks nonlinear predictor influence on the target variable.

---

## Output

The system automatically generates:

- High-resolution visualizations
- Structured analytical PDF report
- Cluster silhouette score evaluation
- Feature importance ranking
- Risk distribution profiling

---

## Use Cases

- Clinical risk modeling
- Population stratification
- Data-driven triage systems
- Academic research analysis
- Machine learning portfolio demonstration
- Automated reporting pipelines

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- ReportLab

---

## Author

Sagnik  
GitHub: https://github.com/sagnik10

---

## License

Released under the MIT License.

---

## Future Enhancements

- Model validation metrics (ROC, AUC, confusion matrices)
- Statistical inference modules
- Automated executive summary generation
- Journal-style report formatting
- Dockerized deployment
- CI/CD integration
- CLI-based execution mode

---

## Disclaimer

This analytical system is intended for research and educational purposes.  
Clinical or operational decisions should not rely solely on automated outputs without domain expert validation.


<!-- SAGNIK10-STYLE-FOOTER -->
---
<p align="center">
  Maintained by <a href="https://github.com/sagnik10"><strong>Sagnik Sen</strong></a> ·
  <a href="https://sagnik10.github.io/SagnikSenPortfolio/">Portfolio &amp; 1-on-1 Teaching</a> ·
  <a href="https://www.linkedin.com/in/sagnik-sen-0a70431a8/">LinkedIn</a>
</p>
<!-- /SAGNIK10-STYLE-FOOTER -->