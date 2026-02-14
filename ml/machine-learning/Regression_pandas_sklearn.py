# Databricks notebook source
# MAGIC %md
# MAGIC # Regression with scikit-learn and Pandas
# MAGIC This notebook demonstrates regression using scikit-learn and pandas

# COMMAND ----------

#Write Your Email with which account is created
Username="meetraj.solanki@kenexai.com"

# COMMAND ----------

import pandas as pd
import numpy as np


# COMMAND ----------

# Load data 
data_path = "dataset.csv" #train.csv
df = pd.read_csv(data_path,index_col=0)
df = df.drop_duplicates()
df = df.dropna(subset=["target"])
df.head()

# COMMAND ----------

# Define target column

TARGET = "target"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# COMMAND ----------

# Identify categorical and numerical columns

categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(exclude=["object"]).columns

# COMMAND ----------

# Build preprocessing pipeline

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import tempfile
import os
import matplotlib.pyplot as plt
def plot_time_series_demand():
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    age_imputed = imputer.fit_transform(X_train[["age"]])
    age_scaled = scaler.fit_transform(age_imputed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Before scaling
    axes[0].hist(X_train["age"].dropna(), bins=30)
    axes[0].set_title("Before Scaling")
    axes[0].set_xlabel("age")

    # After scaling
    axes[1].hist(age_scaled.flatten(), bins=30)
    axes[1].set_title("After Standard Scaling")
    axes[1].set_xlabel("scaled age")

    plt.tight_layout()
    return fig


# COMMAND ----------

# Create Linear Regression model

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(n_jobs=-1) 

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", lr_model)
])

pipeline.fit(X_train, y_train)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

preds = pipeline.predict(X_test)
lr_mae = mean_absolute_error(y_test, preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, preds))
lr_r2 = r2_score(y_test, preds)

print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))  # Manual square root
print("R2:", r2_score(y_test, preds))

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

X_signature = X_train.copy()

for col in X_signature.select_dtypes(include=["int64", "int32"]).columns:
    X_signature[col] = X_signature[col].astype("float64")

signature = infer_signature(
    X_signature,
    pipeline.predict(X_signature)
)

experiment_name = f"/Users/{Username}/customer_profile_features"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass  
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="lr_regression") as run:
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mae",lr_mae)
    mlflow.log_metric("rmse", lr_rmse)
    mlflow.log_metric("r2", lr_r2)
    fig = plot_time_series_demand()
    mlflow.log_figure(
        fig,
        "age_scaling_distribution.png"
    )
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        signature=signature,
        registered_model_name="customer_profile_score_lr_model"
    )
    run_id = run.info.run_id  

model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
plt.close()

# COMMAND ----------

print(model_uri)

# COMMAND ----------

def predict_price(input_df: pd.DataFrame):
    """
    Serving-style prediction function
    """
    return loaded_model.predict(input_df)


# COMMAND ----------

print(X_test.iloc[:3])
print("\n")
print(y_test.iloc[:3])

# COMMAND ----------

print(predict_price(X_test.iloc[:3]))

# COMMAND ----------

import mlflow.sklearn

model = mlflow.sklearn.load_model(
    "models:/workspace.default.customer_profile_score_lr_model/1"
)

preds = model.predict(X_test.iloc[:3])
print(preds)

# COMMAND ----------

# MAGIC %md
# MAGIC Random Forest Regression

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(n_estimators=50, n_jobs=-1) 

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", random_forest_model)
])

pipeline.fit(X_train, y_train)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

preds = pipeline.predict(X_test)
random_forest_mae = mean_absolute_error(y_test, preds)
random_forest_rmse = np.sqrt(mean_squared_error(y_test, preds))
random_forest_r2 = r2_score(y_test, preds)
print("MAE:", random_forest_mae)
print("RMSE:", random_forest_rmse)  # Manual square root
print("R2:", random_forest_r2)

# COMMAND ----------


import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

X_signature = X_train.copy()

for col in X_signature.select_dtypes(include=["int64", "int32"]).columns:
    X_signature[col] = X_signature[col].astype("float64")

signature = infer_signature(
    X_signature,
    pipeline.predict(X_signature)
)

experiment_name = f"/Users/{Username}/customer_profile_features"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass  
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="random_forest_regression") as run:
    mlflow.log_param("model", "RandomForestRegression")
    mlflow.log_metric("mae",random_forest_mae)
    mlflow.log_metric("rmse", random_forest_rmse)
    mlflow.log_metric("r2", random_forest_r2)
    fig = plot_time_series_demand()
    mlflow.log_figure(
        fig,
        "age_scaling_distribution.png"
    )
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        signature=signature,
        registered_model_name="customer_profile_score_random_forest_regression_model"
    )
    run_id = run.info.run_id  

model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
plt.close()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

def compare_metrics():
    metrics = ["RMSE", "MAE", "R²"]

    Linear_Regression = [lr_rmse, lr_mae, lr_r2] 
    Random_Forest_Regression = [random_forest_rmse, random_forest_mae, random_forest_r2]   

    x = np.arange(len(metrics))
    width = 0.35

    fig = plt.figure(figsize=(8, 5))

    plt.bar(x - width/2, Linear_Regression, width, label="Linear_Regression", color="blue")
    plt.bar(x + width/2, Random_Forest_Regression, width, label="Random_Forest_Regression", color="red")

    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.title("Regression Metrics Comparison")
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()

    # Save PNG
    plt.savefig("regression_metrics_comparison.png")
    plt.show()
    plt.close()
    return fig


# COMMAND ----------

mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name="compare_regression") as run:
    fig = compare_metrics()
    mlflow.log_figure(
        fig,
        "regression_metrics_comparison.png"
    )

# COMMAND ----------


