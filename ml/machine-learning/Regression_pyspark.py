# Databricks notebook source
# MAGIC %md
# MAGIC # Regression with PySpark MLlib
# MAGIC This notebook demonstrates regression using PySpark instead of pandas + scikit-learn

# COMMAND ----------

USERNAME="meetraj.solanki@kenexai.com"

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

# Load data using Spark
data_path = "/Volumes/workspace/default/rawdata/dataset.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Drop duplicates and rows with null target
df = df.dropDuplicates()
df = df.dropna(subset=["target"])
df = df.drop("_c0")
# Display first few rows
display(df.limit(5))

# COMMAND ----------

# Define target column
TARGET = "target"

# Identify categorical and numerical columns
categorical_cols = [field.name for field in df.schema.fields 
                   if isinstance(field.dataType, StringType) and field.name != TARGET]

numerical_cols = [field.name for field in df.schema.fields 
                 if isinstance(field.dataType, (IntegerType, LongType, FloatType, DoubleType)) 
                 and field.name != TARGET and field.name != "_c0"]

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# COMMAND ----------

# Train-test split (80-20)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Pipeline

# COMMAND ----------

# Build preprocessing pipeline
stages = []

# Impute numerical features
num_imputer = Imputer(
    inputCols=numerical_cols,
    outputCols=[f"{c}_imputed" for c in numerical_cols],
    strategy="median"
)
stages.append(num_imputer)

# Scale numerical features
num_assembler = VectorAssembler(
    inputCols=[f"{c}_imputed" for c in numerical_cols],
    outputCol="numerical_features_vec"
)
stages.append(num_assembler)

scaler = StandardScaler(
    inputCol="numerical_features_vec",
    outputCol="numerical_features_scaled",
    withStd=True,
    withMean=True
)
stages.append(scaler)

# Handle categorical features
if categorical_cols:
    # String indexing
    indexers = [StringIndexer(
        inputCol=col,
        outputCol=f"{col}_indexed",
        handleInvalid="keep"  # Similar to sklearn's handle_unknown='ignore'
    ) for col in categorical_cols]
    stages.extend(indexers)
    
    # One-hot encoding
    encoders = [OneHotEncoder(
        inputCol=f"{col}_indexed",
        outputCol=f"{col}_encoded"
    ) for col in categorical_cols]
    stages.extend(encoders)
    
    # Assemble all features
    feature_cols = ["numerical_features_scaled"] + [f"{col}_encoded" for col in categorical_cols]
else:
    feature_cols = ["numerical_features_scaled"]

# Final assembler
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)
stages.append(assembler)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression Model

# COMMAND ----------

# Create Linear Regression model
lr = LinearRegression(
    featuresCol="features",
    labelCol=TARGET,
    predictionCol="prediction"
)

# Build pipeline
lr_pipeline = Pipeline(stages=stages + [lr])

# Train model
lr_model = lr_pipeline.fit(train_df)

# Make predictions
lr_predictions = lr_model.transform(test_df)

# COMMAND ----------

pipeline = Pipeline(stages=stages)
for i, stage in enumerate(stages):
    print(f"{i+1}. {stage.__class__.__name__}")

# COMMAND ----------

# Evaluate Linear Regression
lr_evaluator_rmse = RegressionEvaluator(
    labelCol=TARGET,
    predictionCol="prediction",
    metricName="rmse"
)
lr_evaluator_mae = RegressionEvaluator(
    labelCol=TARGET,
    predictionCol="prediction",
    metricName="mae"
)
lr_evaluator_r2 = RegressionEvaluator(
    labelCol=TARGET,
    predictionCol="prediction",
    metricName="r2"
)

lr_rmse = lr_evaluator_rmse.evaluate(lr_predictions)
lr_mae = lr_evaluator_mae.evaluate(lr_predictions)
lr_r2 = lr_evaluator_r2.evaluate(lr_predictions)

print(f"Linear Regression Metrics:")
print(f"MAE: {lr_mae}")
print(f"RMSE: {lr_rmse}")
print(f"R2: {lr_r2}")

# COMMAND ----------

lr_params={}
for param, value in lr.extractParamMap().items():
    print(f"{param.name}: {value}")
    lr_params[param.name] = value

# COMMAND ----------

def plot_scaling_distribution(train_df):
    # Just collect the data and do scaling in pandas/numpy
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    # Convert to pandas for plotting
    age_data = train_df.select("age").toPandas()
    
    # Use sklearn for visualization (not cached in Spark)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    age_imputed = imputer.fit_transform(age_data[['age']])
    age_scaled = scaler.fit_transform(age_imputed)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(age_data['age'].dropna(), bins=30)
    axes[0].set_title("Before Scaling")
    axes[0].set_xlabel("age")
    
    axes[1].hist(age_scaled.flatten(), bins=30)
    axes[1].set_title("After Standard Scaling")
    axes[1].set_xlabel("scaled age")
    
    plt.tight_layout()
    return fig

# COMMAND ----------

# Log Linear Regression to MLflow
from mlflow.models.signature import infer_signature
import os

spark_input = train_df.limit(5).drop(TARGET)
spark_output = lr_model.transform(spark_input).select("prediction")
signature = infer_signature(spark_input, spark_output)
input_example = (
    train_df
    .limit(5)
    .toPandas()
    .drop(columns=["target"])
)

os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/default/rawdata"

experiment_name = f"/Users/{USERNAME}/customer_profile_features"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="lr_regression_spark") as run:
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mae", lr_mae)
    mlflow.log_metric("rmse", lr_rmse)
    mlflow.log_metric("r2", lr_r2)
    mlflow.log_params(lr_params)
    # Log scaling distribution plot
    fig = plot_scaling_distribution(train_df)
    mlflow.log_figure(fig, "age_scaling_distribution.png")
    plt.close(fig)
    
    # Log model
    mlflow.spark.log_model(
        lr_model,
        artifact_path="model",
        signature=signature,
        registered_model_name="customer_profile_score_lr_model_spark"
    )
    
    lr_run_id = run.info.run_id

print(f"Model URI: runs:/{lr_run_id}/model")

# COMMAND ----------

# Display sample predictions
display(lr_predictions.select(TARGET, "prediction").limit(3))

# COMMAND ----------

import mlflow.spark

new_data = spark.createDataFrame([
    {'age': 35.0, 'income': 60000.0, 'city': 'NY', 'education': 'Masters'},
    {'age': 45.0, 'income': None, 'city': 'LA', 'education': 'PhD'},
])

predictions = lr_model.transform(new_data).select("prediction")
display(predictions)

model = mlflow.spark.load_model(
    "models:/customer_profile_score_lr_model_spark/1"
)
predictions = model.transform(new_data).select("prediction")
display(predictions)


# COMMAND ----------


