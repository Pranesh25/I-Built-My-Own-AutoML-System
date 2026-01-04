# I-Built-My-Own-AutoML-System
I Built My Own AutoML System (And Learned More Than From Any Single ML Model) Most machine learning tutorials teach you how to train a model.  But real-world ML is not about one model - it's about deciding which model to use.

############################
🚀 AutoML-Lite Platform
📌 Overview
AutoML-Lite is a lightweight, end-to-end automated machine learning system for tabular datasets.
It automatically detects the machine learning task, preprocesses data, trains multiple models, evaluates them, and selects the best model using both performance and production-aware logic.

This project focuses on correct ML engineering practices, not just achieving high accuracy.

🎯 Objective
The goal of this project is to:

Build a generic ML pipeline that works on unknown datasets
Reduce manual preprocessing errors
Prevent data leakage
Handle mixed data types safely
Select models intelligently instead of blindly
🔍 What This Project Does
Given a CSV dataset and a target column, the system:

Cleans the dataset
Detects whether the task is classification or regression
Preprocesses numeric and categorical features correctly
Trains multiple suitable ML models
Evaluates models using appropriate metrics
Selects the best model using performance + practicality
Produces a deployable ML pipeline
🔁 System Flow (High Level)
Load CSV ↓ Select Target Column ↓ Drop Rows with Missing Target ↓ Detect Task Type (Classification / Regression) ↓ Feature Profiling ├─ Numeric Features └─ Categorical Features └─ Drop High-Cardinality Columns ↓ Preprocessing Pipeline ├─ Numeric: Impute + Scale └─ Categorical: Impute + Encode (Sparse) ↓ Train-Test Split ↓ Train Multiple Models ↓ Evaluate Models ↓ Production-Aware Model Selection ↓ Final Trained Pipeline

⚙️ Data Preprocessing Strategy
Numeric Features
Missing values handled using median imputation
Features scaled using StandardScaler
Prevents dominance of large numeric values
Categorical Features
Missing values filled with most frequent category
Encoded using One-Hot Encoding
Sparse matrix output for memory efficiency
Rare categories grouped using min_frequency
High-Cardinality Protection
Categorical features with more than 50 unique values are dropped
Prevents dimensional explosion and memory crashes
🤖 Models Used
Classification
Logistic Regression
Stable
Interpretable
Outputs probabilities
Linear Support Vector Machine (LinearSVC)
Strong performance on sparse, high-dimensional data
Regression
Ridge Regression
Linear Support Vector Regression (SVR)
Models are selected dynamically based on detected task type.

📊 Evaluation Metrics
Classification
Accuracy
Weighted F1-score (Primary metric)
F1-score is preferred because it balances precision and recall and handles class imbalance better than accuracy.

Regression
Root Mean Squared Error (RMSE)
R² Score
🧠 Model Selection Logic
Rank models by primary evaluation metric
Select top-N performing models
If multiple models perform similarly:
Prefer models that provide probability outputs
Prefer models that are more stable and production-friendly
This mirrors real-world ML decision-making.

🚫 Common Pitfalls Avoided
No preprocessing before train-test split (prevents data leakage)
No imputation of target labels
No dense conversion of sparse matrices
No accuracy-only optimization
No dataset-specific hardcoding
🛠️ Tech Stack
Python
Pandas, NumPy
scikit-learn
Jupyter Notebook
📌 Future Enhancements
Convert logic into an AutoMLLite class
Add cross-validation and stability scoring
Add dataset profiling summary
Build Streamlit UI
Add experiment tracking with MLflow
Deploy as an inference API
📄 Resume Summary
Built a custom AutoML system that automatically preprocesses datasets, detects task type, evaluates multiple machine learning models, applies production-aware selection logic, and outputs a deployable trained pipeline.

✅ Project Status
✔ Core AutoML engine completed
✔ Memory-safe preprocessing
✔ Dynamic model selection
✔ Production-aware decision logic
