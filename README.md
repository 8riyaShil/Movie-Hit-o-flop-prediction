# 🎬 Movie Hit or Flop Prediction using Decision Tree

## 📌 Project Overview

This project uses **Machine Learning** to predict whether a movie will be a **Hit or Flop** based on different movie-related features.

A **Decision Tree Classifier** is trained to perform binary classification, where:

* `0` = Flop
* `1` = Hit

The project covers the complete machine learning workflow, including data loading, data inspection, missing-value handling, data filtering, duplicate removal, train-test splitting, model training, prediction, and performance evaluation.

---

## 🎯 Objective

The main objective of this project is to build a machine learning model that can predict a movie's success based on factors such as:

* Movie budget
* Marketing expenditure
* Number of screens
* Star power
* Genre
* Runtime
* Critic score

The target variable is:

```text
is_hit
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Data visualization
* **Scikit-learn** – Machine Learning

### Machine Learning Algorithm

**Decision Tree Classifier**

---

## 📂 Dataset

The project uses the following CSV dataset:

```text
riya_shil_movie_hit_or_flop_prediction.csv
```

The dataset contains **2,000 records and 8 columns** before preprocessing.

### Dataset Features

| Feature                   | Description                         |
| ------------------------- | ----------------------------------- |
| `budget_million`          | Movie budget in millions            |
| `marketing_spend_million` | Marketing expenditure in millions   |
| `num_screens`             | Number of screens showing the movie |
| `star_power_score`        | Star power score                    |
| `genre_code`              | Numerical genre representation      |
| `runtime_minutes`         | Movie runtime in minutes            |
| `critic_score`            | Critic score                        |
| `is_hit`                  | Target variable: 0 = Flop, 1 = Hit  |

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Missing Value Analysis
   ↓
Missing Value Imputation
   ↓
Exploratory Data Analysis
   ↓
Invalid/Unrealistic Value Filtering
   ↓
Duplicate Removal
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Decision Tree Training
   ↓
Prediction
   ↓
Model Evaluation
```

---

## 🔍 Data Exploration

The dataset was initially examined using:

```python
df.describe()
df.info()
```

The project also checks:

* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate rows

The original dataset contained:

```text
2000 rows
8 columns
```

There were missing values in several numerical features and **5 duplicate rows** were detected during the initial inspection.

---

## 🧹 Data Preprocessing

### 1. Missing Value Handling

Missing values in the numerical feature columns were filled using the **median** of each column.

```python
numeric_columns = df.columns.drop('is_hit')

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())
```

---

### 2. Data Filtering

The dataset was filtered to remove values outside the specified ranges.

The following conditions were applied:

```text
Budget: 0–500 million
Marketing Spend: 0–150 million
Number of Screens: 0–6000
Star Power Score: >= 0
Genre Code: >= 0
Runtime: >= 0
Critic Score: >= 0
```

This step was used to remove unrealistic or invalid values from the dataset.

---

### 3. Duplicate Removal

Duplicate records were removed using:

```python
df = df.drop_duplicates()
```

After preprocessing and duplicate removal, the dataset contained:

```text
1850 rows
8 columns
```

---

## 📊 Exploratory Data Analysis

Scatter plots were created to examine the relationship between each input feature and the target variable `is_hit`.

The project uses:

```python
matplotlib
seaborn
```

for visualization.

The visualizations help examine how movie characteristics relate to whether a movie is classified as a hit or flop.

---

## 🔀 Feature and Target Separation

The input features were separated from the target variable:

```python
X = df.drop('is_hit', axis=1)
y = df['is_hit']
```

### Features

```text
budget_million
marketing_spend_million
num_screens
star_power_score
genre_code
runtime_minutes
critic_score
```

### Target

```text
is_hit
```

---

## ✂️ Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Dataset Split

| Dataset  | Samples |
| -------- | ------: |
| Training |    1480 |
| Testing  |     370 |

`random_state=42` was used to make the experiment reproducible.

`stratify=y` was used to maintain the class distribution between the training and testing datasets.

---

## 🌳 Decision Tree Model

A **Decision Tree Classifier** was used for prediction.

The model was configured with a maximum tree depth of `4`.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=4
)

model.fit(X_train, y_train)
```

The trained model was then used to predict the test data.

```python
y_pred = model.predict(X_test)
```

---

## 📈 Model Evaluation

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Decision Tree Visualization

---

## 🎯 Accuracy

The model achieved:

```text
Accuracy: 93.51%
```

This means that the model correctly classified approximately **93.51% of the movies in the test dataset**.

---

## 📊 Confusion Matrix

The obtained confusion matrix was:

```text
[[185   5]
 [ 19 161]]
```

The matrix can be interpreted as:

|                 | Predicted Flop | Predicted Hit |
| --------------- | -------------: | ------------: |
| **Actual Flop** |            185 |             5 |
| **Actual Hit**  |             19 |           161 |

### Interpretation

* **185** Flop movies were correctly predicted as Flop.
* **161** Hit movies were correctly predicted as Hit.
* **5** Flop movies were incorrectly predicted as Hit.
* **19** Hit movies were incorrectly predicted as Flop.

---

## 📋 Classification Report

The model produced the following results:

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------: | -----: | -------: | ------: |
| Flop (0)         |      0.91 |   0.97 |     0.94 |     190 |
| Hit (1)          |      0.97 |   0.89 |     0.93 |     180 |
| **Accuracy**     |           |        | **0.94** | **370** |
| Macro Average    |      0.94 |   0.93 |     0.93 |     370 |
| Weighted Average |      0.94 |   0.94 |     0.93 |     370 |

---

## 🌳 Decision Tree Visualization

The trained decision tree was visualized using Scikit-learn's `plot_tree()` function.

```python
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Flop", "Hit"],
    filled=True,
    fontsize=8
)
```

This visualization makes it possible to understand how the Decision Tree makes classification decisions based on the movie features.

---

## 📁 Project Structure

```text
Movie-Hit-or-Flop-Prediction/
│
├── riya_shil_movie_hit_or_flop_prediction.ipynb
├── riya_shil_movie_hit_or_flop_prediction.csv
├── README.md
└── images/
    ├── data_visualization.png
    └── decision_tree.png
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Open the Jupyter Notebook

Open:

```text
riya_shil_movie_hit_or_flop_prediction.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

### 4. Place the Dataset

Make sure the CSV file is available:

```text
riya_shil_movie_hit_or_flop_prediction.csv
```

If you are running the notebook locally, update the CSV file path accordingly.

### 5. Run the Notebook

Run all cells sequentially to:

1. Load the dataset
2. Explore the data
3. Handle missing values
4. Filter invalid values
5. Remove duplicates
6. Split the dataset
7. Train the Decision Tree
8. Generate predictions
9. Evaluate the model
10. Visualize the Decision Tree

---

## 📌 Results Summary

| Metric                      |                   Result |
| --------------------------- | -----------------------: |
| Original Dataset            |                2000 rows |
| Dataset After Preprocessing |                1850 rows |
| Training Samples            |                     1480 |
| Testing Samples             |                      370 |
| Model                       | Decision Tree Classifier |
| Maximum Depth               |                        4 |
| Accuracy                    |               **93.51%** |

---

## 🔮 Future Improvements

The project can be further improved by:

* Hyperparameter tuning using `GridSearchCV`
* Comparing Decision Tree with Random Forest
* Testing Logistic Regression, SVM, KNN, and other classifiers
* Performing cross-validation
* Adding feature importance analysis
* Improving visualization
* Creating a prediction interface using Streamlit
* Deploying the model as a web application
* Evaluating additional performance metrics such as ROC-AUC

---

## 👩‍💻 Author

**Riya Shil**

BCA Student | Aspiring AI/ML & Data Science Professional

---

## ⭐ Conclusion

This project demonstrates an end-to-end **Machine Learning classification workflow** for predicting whether a movie will be a **Hit or Flop**.

Using a **Decision Tree Classifier**, the project achieved an accuracy of **93.51%** on the test dataset. The project also demonstrates important machine learning concepts such as data preprocessing, exploratory data analysis, train-test splitting, classification, confusion matrix analysis, and model visualization.

If you found this project useful, consider giving the repository a ⭐ on GitHub.
