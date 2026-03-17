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

# load the files 
file_riverside_1990 = "/content/drive/MyDrive/urban_expansion_exports/riverside_1990_training_samples.csv"
file_riverside_2000 = "/content/drive/MyDrive/urban_expansion_exports/riverside_2000_training_samples.csv"
file_riverside_2010 = "/content/drive/MyDrive/urban_expansion_exports/riverside_2010_training_samples.csv"
file_riverside_2020 = "/content/drive/MyDrive/urban_expansion_exports/riverside_2020_training_samples.csv"

# optional holdout / inspection only
file_austin_2020 = "/content/drive/MyDrive/urban_expansion_exports/austin_2020_training_samples.csv"

df_riverside_1990 = spark.read.csv(file_riverside_1990, header=True, inferSchema=True)
df_riverside_2000 = spark.read.csv(file_riverside_2000, header=True, inferSchema=True)
df_riverside_2010 = spark.read.csv(file_riverside_2010, header=True, inferSchema=True)
df_riverside_2020 = spark.read.csv(file_riverside_2020, header=True, inferSchema=True)
df_austin_2020 = spark.read.csv(file_austin_2020, header=True, inferSchema=True)

# defining features and labels 
feature_cols = ["red", "green", "blue", "nir", "ndvi"]
target_col = "label"
required_cols = feature_cols + [target_col]

# helper for cleaning the data 
def clean_df(df_in, city_name=None, year_val=None):
    df_out = df_in

    # standardize city/year if missing or inconsistent
    if city_name is not None and "city" in df_out.columns:
        df_out = df_out.withColumn("city", lit(city_name))
    if year_val is not None and "year" in df_out.columns:
        df_out = df_out.withColumn("year", lit(year_val))

    # drop rows missing required columns
    df_out = df_out.dropna(subset=required_cols)

    # cast features
    for c in feature_cols:
        df_out = df_out.withColumn(c, col(c).cast("double"))

    # cast label as integer-like numeric
    df_out = df_out.withColumn(target_col, col(target_col).cast("double"))

    # keep only binary labels 0/1
    df_out = df_out.filter(col(target_col).isin([0.0, 1.0]))

    # drop remaining nulls / duplicates
    df_out = df_out.dropna(subset=required_cols)
    df_out = df_out.dropDuplicates()

    return df_out

df_riverside_1990 = clean_df(df_riverside_1990, "riverside", 1990)
df_riverside_2000 = clean_df(df_riverside_2000, "riverside", 2000)
df_riverside_2010 = clean_df(df_riverside_2010, "riverside", 2010)
df_riverside_2020 = clean_df(df_riverside_2020, "riverside", 2020)
df_austin_2020 = clean_df(df_austin_2020, "austin", 2020)

# combined inspection dataframe 
df_all = (
    df_riverside_1990
    .unionByName(df_riverside_2000)
    .unionByName(df_riverside_2010)
    .unionByName(df_riverside_2020)
    .unionByName(df_austin_2020)
)

print("=== Schema ===")
df_all.printSchema()

print("=== Sample Rows ===")
df_all.show(5, truncate=False)

print("=== Cleaned Row Count ===")
print(df_all.count())

print("=== Label Distribution by City and Year ===")
df_all.groupBy("city", "year", "label").count().orderBy("city", "year", "label").show()

# optional feature statistics 
for yr in [1990, 2000, 2010, 2020]:
    print(f"=== Riverside Feature Stats for Year {yr} ===")
    yr_df = df_all.filter((col("city") == "riverside") & (col("year") == yr))
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

# training data (but for riverside region only)
legacy_df = (
    df_riverside_1990
    .unionByName(df_riverside_2000)
    .unionByName(df_riverside_2010)
)

landsat8_df = df_riverside_2020

print("=== Legacy Data Row Count (Riverside 1990/2000/2010) ===")
print(legacy_df.count())

print("=== 2020 Data Row Count (Riverside 2020 only) ===")
print(landsat8_df.count())

print("=== Legacy Label Distribution ===")
legacy_df.groupBy("label").count().orderBy("label").show()

print("=== Riverside 2020 Label Distribution ===")
landsat8_df.groupBy("label").count().orderBy("label").show()

# balance by downsampling 
# safer than aggressive oversampling for your case
def balance_binary_downsample(df_in, label_col="label", seed=42):
    class0 = df_in.filter(col(label_col) == 0.0)
    class1 = df_in.filter(col(label_col) == 1.0)

    n0 = class0.count()
    n1 = class1.count()

    print("Current class counts:")
    print(f"  class 0 = {n0}")
    print(f"  class 1 = {n1}")

    if n0 == 0 or n1 == 0:
        print("WARNING: One class is missing. Returning original dataframe.")
        return df_in

    target_n = builtins.min(n0, n1)

    if n0 > target_n:
        frac0 = target_n / n0
        class0 = class0.sample(withReplacement=False, fraction=frac0, seed=seed)

    if n1 > target_n:
        frac1 = target_n / n1
        class1 = class1.sample(withReplacement=False, fraction=frac1, seed=seed)

    balanced = class0.unionByName(class1)
    return balanced

# balancing the training sets 
print("=== Balancing Legacy Data ===")
legacy_balanced_df = balance_binary_downsample(legacy_df, label_col="label", seed=42)
legacy_balanced_df.groupBy("label").count().orderBy("label").show()

print("=== Balancing Riverside 2020 Data ===")
landsat8_balanced_df = balance_binary_downsample(landsat8_df, label_col="label", seed=42)
landsat8_balanced_df.groupBy("label").count().orderBy("label").show()

# splitting into training and validation split 
legacy_train_df, legacy_val_df = legacy_balanced_df.randomSplit([0.8, 0.2], seed=42)
landsat8_train_df, landsat8_val_df = landsat8_balanced_df.randomSplit([0.8, 0.2], seed=42)

print("=== Legacy train/val counts ===")
print(legacy_train_df.count(), legacy_val_df.count())

print("=== 2020 train/val counts ===")
print(landsat8_train_df.count(), landsat8_val_df.count())

# training the random forest model 
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
        numTrees=80,
        maxDepth=8,
        minInstancesPerNode=10,
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, rf])

    print(f"=== Training {model_name} ===")
    model = pipeline.fit(train_df)

    model.write().overwrite().save(model_path)
    print(f"=== Saved {model_name} to: {model_path} ===")

    return model

# evaluate model 
def evaluate_model(model, eval_df, model_name):
    preds = model.transform(eval_df)

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    acc = evaluator_acc.evaluate(preds)
    f1 = evaluator_f1.evaluate(preds)
    precision = evaluator_precision.evaluate(preds)
    recall = evaluator_recall.evaluate(preds)
    # print results 
    print(f"=== Validation Metrics: {model_name} ===")
    print(f"Accuracy           = {acc:.4f}")
    print(f"F1 Score           = {f1:.4f}")
    print(f"Weighted Precision = {precision:.4f}")
    print(f"Weighted Recall    = {recall:.4f}")

    print("=== Confusion-style table ===")
    preds.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# train legacy model 
legacy_model_path = "data/urban_rf_model_legacy"
legacy_model = train_rf_model(
    legacy_train_df,
    feature_cols,
    legacy_model_path,
    "Legacy Model (Riverside 1990/2000/2010)"
)

evaluate_model(legacy_model, legacy_val_df, "Legacy Model")

# train 2020 model 
landsat8_model_path = "data/urban_rf_model_2020"
landsat8_model = train_rf_model(
    landsat8_train_df,
    feature_cols,
    landsat8_model_path,
    "2020 Model (Riverside 2020 only)"
)

evaluate_model(landsat8_model, landsat8_val_df, "2020 Model")

# print feature importance 
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