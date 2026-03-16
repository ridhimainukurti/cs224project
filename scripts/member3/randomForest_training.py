from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


# creates the spark session 
spark = SparkSession.builder \
    .appName("UrbanExpansionRandomForest") \
    .getOrCreate()

file_riverside_1990 = "riverside_1990_training_samples.csv"
file_riverside_2000 = "riverside_2000_training_samples.csv"
file_riverside_2010 = "riverside_2010_training_samples.csv"
file_riverside_2020 = "riverside_2020_training_samples.csv"
file_austin_2020 = "austin_2020_training_samples.csv"

# reading the csv files for training 
df_1990 = spark.read.csv(file_riverside_1990, header=True, inferSchema=True)
df_2000 = spark.read.csv(file_riverside_2000, header=True, inferSchema=True)
df_2010 = spark.read.csv(file_riverside_2010, header=True, inferSchema=True)
df_2020 = spark.read.csv(file_riverside_2020, header=True, inferSchema=True)
df_austin_2020 = spark.read.csv(file_austin_2020, header=True, inferSchema=True)

df = (
    df_1990
    .unionByName(df_2000)
    .unionByName(df_2010)
    .unionByName(df_2020)
    .unionByName(df_austin_2020)
)

# print schema to confirm column names and data types 
print("=== Schema ===")
df.printSchema()

# show a few example rows 
print("=== Sample Rows ===")
df.show(5, truncate=False)

# print the total number of raw rows before cleaning 
print("=== Raw Row Count ===")
print(df.count())

# define features and labels 
feature_cols = ["red", "green", "blue", "nir", "ndvi"]
target_col = "label"
required_cols = feature_cols + [target_col]

# clean the data 
df_clean = df.dropna(subset=required_cols)

for c in feature_cols:
    df_clean = df_clean.withColumn(c, col(c).cast("double"))

df_clean = df_clean.withColumn(target_col, col(target_col).cast("double"))
df_clean = df_clean.dropna(subset=required_cols)
df_clean = df_clean.dropDuplicates()

print("=== Cleaned Row Count ===")
print(df_clean.count())

# summaries to verify whether classes are balanced
print("=== Label Distribution ===")
df_clean.groupBy("label").count().orderBy("label").show()

print("=== Label Distribution by Year ===")
df_clean.groupBy("year", "label").count().orderBy("year", "label").show()

print("=== Label Distribution by City and Year ===")
df_clean.groupBy("city", "year", "label").count().orderBy("city", "year", "label").show()

print("=== Subclass Distribution by Year ===")
if "subclass" in df_clean.columns:
    df_clean.groupBy("year", "subclass").count().orderBy("year", "subclass").show()

# print descriptive statistics for each spectral feature by year 
# useful for checking wehther feature ranges differ alot 
for yr in [1990, 2000, 2010, 2020]:
    print(f"=== Feature Stats for Year {yr} ===")
    yr_df = df_clean.filter(col("year") == yr)
    yr_df.select(
        mean("red").alias("red_mean"),
        stddev("red").alias("red_std"),
        min("red").alias("red_min"),
        max("red").alias("red_max"),
        mean("green").alias("green_mean"),
        stddev("green").alias("green_std"),
        min("green").alias("green_min"),
        max("green").alias("green_max"),
        mean("blue").alias("blue_mean"),
        stddev("blue").alias("blue_std"),
        min("blue").alias("blue_min"),
        max("blue").alias("blue_max"),
        mean("nir").alias("nir_mean"),
        stddev("nir").alias("nir_std"),
        min("nir").alias("nir_min"),
        max("nir").alias("nir_max"),
        mean("ndvi").alias("ndvi_mean"),
        stddev("ndvi").alias("ndvi_std"),
        min("ndvi").alias("ndvi_min"),
        max("ndvi").alias("ndvi_max"),
    ).show(truncate=False)

# spliting data into to get trained under two models
# Legacy model = 1990, 2000, 2010
legacy_df = df_clean.filter(col("year").isin([1990, 2000, 2010]))

# 2020 model = 2020 only
landsat8_df = df_clean.filter(col("year") == 2020)

print("=== Legacy Data Row Count (1990/2000/2010) ===")
print(legacy_df.count())

print("=== 2020 Data Row Count ===")
print(landsat8_df.count())

print("=== Legacy Label Distribution ===")
legacy_df.groupBy("label").count().orderBy("label").show()

print("=== 2020 Label Distribution ===")
landsat8_df.groupBy("label").count().orderBy("label").show()

print("=== Legacy Label Distribution by Year ===")
legacy_df.groupBy("year", "label").count().orderBy("year", "label").show()

print("=== 2020 Label Distribution by City ===")
landsat8_df.groupBy("city", "label").count().orderBy("city", "label").show()

# helps balance an imbalanced binary dataset 
def rebalance_binary(df_in, label_col="label", minority_label=1.0, majority_label=0.0, seed=42):
    minority_df = df_in.filter(col(label_col) == minority_label)
    majority_df = df_in.filter(col(label_col) == majority_label)

    # count each class 
    minority_count = minority_df.count()
    majority_count = majority_df.count()

    print("Current class counts:")
    print(f"  minority ({minority_label}) = {minority_count}")
    print(f"  majority ({majority_label}) = {majority_count}")

    # if minority class is smaller, oversample it
    if minority_count < majority_count:
        oversample_ratio = majority_count / minority_count
        minority_oversampled = minority_df.sample(
            withReplacement=True,
            fraction=oversample_ratio,
            seed=seed
        )
        balanced_df = majority_df.unionByName(minority_oversampled)
    else:
        balanced_df = df_in

    return balanced_df

# rebalance legacy data 
print("=== Rebalancing Legacy Data ===")
legacy_train_df = rebalance_binary(
    legacy_df,
    label_col="label",
    minority_label=1.0,
    majority_label=0.0,
    seed=42
)

print("=== Legacy Rebalanced Class Counts ===")
legacy_train_df.groupBy("label").count().orderBy("label").show()

# rebalance 2020 data 
print("=== Rebalancing 2020 Data ===")
landsat8_train_df = rebalance_binary(
    landsat8_df,
    label_col="label",
    minority_label=1.0,
    majority_label=0.0,
    seed=42
)

print("=== 2020 Rebalanced Class Counts ===")
landsat8_train_df.groupBy("label").count().orderBy("label").show()

# train the model 
def train_rf_model(train_df, feature_cols, model_path, model_name):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=100,
        maxDepth=12,
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, rf])

    print(f"=== Training {model_name} ===")
    model = pipeline.fit(train_df)

    model.write().overwrite().save(model_path)
    print(f"=== Saved {model_name} to: {model_path} ===")

    return model

# training the legacy model
legacy_model_path = "data/urban_rf_model_legacy"
legacy_model = train_rf_model(
    legacy_train_df,
    feature_cols,
    legacy_model_path,
    "Legacy Model (1990/2000/2010)"
)

# training the 2020 model 
landsat8_model_path = "data/urban_rf_model_2020"
landsat8_model = train_rf_model(
    landsat8_train_df,
    feature_cols,
    landsat8_model_path,
    "2020 Model"
)

# helper for printing the feature importance 
def print_feature_importances(model, feature_cols, model_name):
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray()

    print(f"=== Feature Importances for {model_name} ===")
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"{feat}: {imp:.6f}")

print_feature_importances(legacy_model, feature_cols, "Legacy Model")
print_feature_importances(landsat8_model, feature_cols, "2020 Model")

print("=== Training complete ===")
print(f"Legacy model saved at: {legacy_model_path}")
print(f"2020 model saved at: {landsat8_model_path}")