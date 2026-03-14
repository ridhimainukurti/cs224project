from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# =========================================================
# 1. CREATE SPARK SESSION
# =========================================================
spark = SparkSession.builder \
    .appName("UrbanExpansionRandomForest") \
    .getOrCreate()

# =========================================================
# 2. LOAD CSV FILES FROM MEMBER 2
#    Paths are relative to the project root folder.
# =========================================================
file_1990 = "data/riverside_1990_training_samples.csv"
file_2000 = "data/riverside_2000_training_samples.csv"
file_2010 = "data/riverside_2010_training_samples.csv"
file_2020 = "data/riverside_2020_training_samples.csv"

df_1990 = spark.read.csv(file_1990, header=True, inferSchema=True)
df_2000 = spark.read.csv(file_2000, header=True, inferSchema=True)
df_2010 = spark.read.csv(file_2010, header=True, inferSchema=True)
df_2020 = spark.read.csv(file_2020, header=True, inferSchema=True)

# Merge all years into one Riverside training dataset
df = df_1990.unionByName(df_2000).unionByName(df_2010).unionByName(df_2020)

# =========================================================
# 3. INSPECT SCHEMA AND BASIC STRUCTURE
# =========================================================
print("=== Schema ===")
df.printSchema()

print("=== Sample Rows ===")
df.show(5, truncate=False)

print("=== Row Count ===")
print(df.count())

# =========================================================
# 4. CLEAN DATASET
# =========================================================
feature_cols = ["red", "green", "blue", "nir", "ndvi"]
target_col = "label"
required_cols = feature_cols + [target_col]

# Drop rows with nulls in required columns
df_clean = df.dropna(subset=required_cols)

# Ensure numeric types
for c in feature_cols:
    df_clean = df_clean.withColumn(c, col(c).cast("double"))

df_clean = df_clean.withColumn(target_col, col(target_col).cast("double"))

# Drop rows that became null after casting
df_clean = df_clean.dropna(subset=required_cols)

# Remove duplicates
df_clean = df_clean.dropDuplicates()

print("=== Cleaned Row Count ===")
print(df_clean.count())

# =========================================================
# 5. DATA QUALITY CHECKS
# =========================================================
print("=== Label Distribution ===")
df_clean.groupBy("label").count().show()

print("=== Subclass Distribution ===")
if "subclass" in df_clean.columns:
    df_clean.groupBy("subclass").count().show()

print("=== Year Distribution ===")
if "year" in df_clean.columns:
    df_clean.groupBy("year").count().show()

invalid_labels = df_clean.filter(~col("label").isin([0.0, 1.0]))
print("=== Invalid Label Count ===")
print(invalid_labels.count())

# =========================================================
# 6. FEATURE VECTOR PREPARATION
# =========================================================
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================
train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)

print("=== Train Count ===")
print(train_df.count())

print("=== Test Count ===")
print(test_df.count())

# =========================================================
# 8. TRAIN RANDOM FOREST CLASSIFIER
# =========================================================
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    numTrees=50,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(train_df)

# =========================================================
# 9. MAKE PREDICTIONS
# =========================================================
predictions = model.transform(test_df)

print("=== Prediction Sample ===")
predictions.select(
    "red", "green", "blue", "nir", "ndvi",
    "label", "prediction", "probability"
).show(10, truncate=False)

# =========================================================
# 10. EVALUATE MODEL
# =========================================================
accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

precision = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(predictions)

recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
).evaluate(predictions)

print("=== Evaluation Metrics ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =========================================================
# 11. CONFUSION MATRIX
# =========================================================
print("=== Confusion Matrix ===")
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# =========================================================
# 12. SAVE MODEL
# =========================================================
model_path = "data/urban_rf_model"
model.write().overwrite().save(model_path)
print(f"Model saved to: {model_path}")

spark.stop()
