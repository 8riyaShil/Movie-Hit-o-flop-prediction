# 🎬 Movie Hit or Flop Prediction

A Machine Learning project that predicts whether a movie will be a **Hit** or **Flop** using movie-related features such as budget, marketing spend, number of screens, star power, genre, runtime, and critic score.

The project uses a **Decision Tree Classifier** with Python and Scikit-learn.

---

## 📌 Project Overview

The objective of this project is to build a binary classification model that predicts the success of a movie.

- **0 → Flop**
- **1 → Hit**

The original dataset contains **2,000 records and 8 columns**. After data preprocessing, invalid-value filtering, and duplicate removal, the final dataset contains **1,811 records**.

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `budget_million` | Movie budget in millions |
| `marketing_spend_million` | Marketing expenditure in millions |
| `num_screens` | Number of screens |
| `star_power_score` | Star power score |
| `genre_code` | Encoded movie genre |
| `runtime_minutes` | Movie runtime in minutes |
| `critic_score` | Critic score |
| `is_hit` | Target variable: 0 = Flop, 1 = Hit |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Missing Value Handling
   ↓
Invalid Value & Outlier Filtering
   ↓
Duplicate Removal
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Decision Tree Classifier
   ↓
Prediction
   ↓
Model Evaluation
