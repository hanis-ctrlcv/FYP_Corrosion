# 🛠️ Reinforced Deep Learning Corrosion Prediction Dashboard

This Streamlit-based dashboard predicts **pipeline corrosion rates** and estimates **remaining life** using a hybrid *Reinforced Deep Learning* model that combines **Deep Learning (DL)**, **Random Forest (RF)**, and **XGBoost (XGB)** algorithms.  

Developed as part of the **Final Year Project (FYP)** for the course **Undergraduate Thesis Project (UTP)** — *Universiti Teknologi PETRONAS (UTP)*.

---

## 📘 Project Overview

Corrosion in industrial pipelines can lead to severe safety, environmental, and economic consequences.  
This project introduces an AI-assisted predictive framework to forecast corrosion rates based on environmental and material parameters, and to estimate the **remaining pipeline life** under various operating conditions.

The dashboard provides an **interactive visualization** and **real-time prediction** interface powered by a trained ensemble model.

---

## 🚀 Features

### 🔍 1. Model Performance Visualization
- Compares prediction accuracy of:
  - Deep Learning (DL)
  - Random Forest (RF)
  - XGBoost (XGB)
  - Reinforced Deep Learning (hybrid ensemble)
- Displays R², MAE, and RMSE metrics with visual scatter plots.

### 🧮 2. Pipeline Condition Simulation
- Interactive **PIPE A, PIPE B, PIPE C** panels.
- Real-time corrosion rate visualization (color-coded severity levels).
- Styled corrosion prediction results for each model.

### 📈 3. Remaining Life Estimation
- Deterministic and Monte Carlo–based calculation of **pipeline remaining life**.
- Adjusts for uncertainty using Mean Absolute Error (MAE) and pitting factor.
- Output includes median and percentile range estimates.

### 📊 4. Bulk Prediction
- Upload a CSV dataset for batch corrosion rate predictions.
- Automatically saves results as a downloadable CSV.

### 🌡️ 5. Data Exploration Tools
- Correlation heatmap between key variables.
- Pairplot visualization for feature interaction insights.

---

## 🧠 Machine Learning Models

| Model | Description | Type |
|:------|:-------------|:------|
| Deep Learning | Neural network trained on corrosion data | Regression |
| Random Forest | Tree-based ensemble model | Regression |
| XGBoost | Gradient boosting model | Regression |
| Reinforced Deep Learning | Weighted hybrid of DL + RF + XGB | Ensemble Regression |

---

## 📁 Project Structure

