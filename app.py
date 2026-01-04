## how to run: streamlit run app.py                  #
## pip install streamlit pandas scikit-learn joblib  #
## streamlit run app.py                              #
################# or #################               #
## Use Python to run Streamlit directly              #
# In PowerShell / CMD, run:                          #
# python -m pip install streamlit                    #
# Then run Streamlit like this:                      #
# python -m streamlit run app.py                     #
######################################################

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# IMPORT AutoMLLite
# ---------------------------------
# Option 1: If AutoMLLite is in another file
# from automl_lite import AutoMLLite

# Option 2: If you paste AutoMLLite class above this code
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class AutoMLLite:
    def __init__(self, target_column, max_unique=50, test_size=0.2,
                 random_state=42, cv_folds=5):
        self.target_column = target_column
        self.max_unique = max_unique
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.task_type = None
        self.results_df = None
        self.best_model_name = None
        self.final_pipeline = None
        self.profile_report = {}

    def _detect_task_type(self, y):
        if y.dtype == "object":
            return "classification"
        if y.nunique() <= 10:
            return "classification"
        return "regression"

    def _get_models(self):
        if self.task_type == "classification":
            return {
                "LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1),
                "LinearSVM": LinearSVC()
            }
        else:
            return {
                "RidgeRegression": Ridge(),
                "LinearSVR": LinearSVR()
            }

    def _profile_dataset(self, df):
        self.profile_report = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "target": self.target_column,
            "target_unique_values": df[self.target_column].nunique()
        }

    def _build_preprocessor(self, X):
        cat_cols = X.select_dtypes(include=["object"]).columns
        high_card_cols = [c for c in cat_cols if X[c].nunique() > self.max_unique]
        X = X.drop(columns=high_card_cols)

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        self.profile_report["numeric_features"] = len(num_cols)
        self.profile_report["categorical_features"] = len(cat_cols)
        self.profile_report["dropped_high_cardinality"] = high_card_cols

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                min_frequency=10
            ))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ])

        return X, preprocessor

    def fit(self, df):
        df = df.dropna(subset=[self.target_column])
        self._profile_dataset(df)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        self.task_type = self._detect_task_type(y)
        self.profile_report["task_type"] = self.task_type

        X, preprocessor = self._build_preprocessor(X)
        models = self._get_models()

        results = []
        for name, model in models.items():
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            if self.task_type == "classification":
                cv = cross_val_score(
                    pipe, X, y, cv=self.cv_folds, scoring="f1_weighted"
                )
                score = cv.mean()
                results.append({"model": name, "cv_f1": round(score, 4)})
            else:
                cv = cross_val_score(
                    pipe, X, y, cv=self.cv_folds,
                    scoring="neg_root_mean_squared_error"
                )
                score = -cv.mean()
                results.append({"model": name, "cv_rmse": round(score, 4)})

        self.results_df = pd.DataFrame(results)

        metric = "cv_f1" if self.task_type == "classification" else "cv_rmse"
        self.results_df = self.results_df.sort_values(
            metric, ascending=(self.task_type != "classification")
        )

        self.best_model_name = (
            "LogisticRegression"
            if "LogisticRegression" in self.results_df.head(2)["model"].values
            else self.results_df.iloc[0]["model"]
        )

        self.final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", models[self.best_model_name])
        ])
        self.final_pipeline.fit(X, y)

        return self

    def save(self, path):
        joblib.dump(self.final_pipeline, path)


# ---------------------------------
# STREAMLIT UI
# ---------------------------------

st.set_page_config(page_title="AutoML-Lite", layout="centered")
st.title("🚀 AutoML-Lite Platform")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):
        with st.spinner("Running AutoML..."):
            automl = AutoMLLite(target_column=target)
            automl.fit(df)

        st.success("AutoML Completed!")

        st.subheader("📊 Dataset Profile")
        st.json(automl.profile_report)

        st.subheader("🤖 Model Comparison")
        st.dataframe(automl.results_df)

        st.subheader("🏆 Best Model Selected")
        st.code(automl.best_model_name)

        model_path = "automl_best_model.joblib"
        automl.save(model_path)

        with open(model_path, "rb") as f:
            st.download_button(
                "⬇️ Download Trained Model",
                f,
                file_name=model_path
            )
