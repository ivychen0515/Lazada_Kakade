from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import udf, regexp_replace
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
spark = SparkSession.builder.appName("kakade").getOrCreate()
# function define#####
# count number of tokens in given column
countTokens = udf(lambda words: len(words), IntegerType())
# count total character length in title/description
charCount = udf(lambda x: len(x), IntegerType())
# count number of words w/o special/stop words in title/description
def useful_token_func(col1, col2):
    if len(col1) > col2:
        return 0
    else:
        return col2 - len(col1)
usefulToken = udf(useful_token_func, IntegerType())
# extract unique words from title
uniqueExtract = udf(lambda words: list(set(words)), ArrayType(StringType()))
# label string to int
string2int = udf(lambda x: int(x), IntegerType())
# Add header to data feature
data_schema = [StructField("index", IntegerType(), True), StructField("title", StringType(), True), \
               StructField("new_title", StringType(), True), StructField("description", StringType(), True), \
               StructField("new_des", StringType(), True), StructField("category_1", StringType(), True), \
               StructField("concise", IntegerType(), True), StructField("clarity", IntegerType(), True)]
# country sku_id title category_1 category_2 category_3 short_description price product_type
data_struc = StructType(fields=data_schema)
# load all data
# train_data=spark.read.csv("/home/kakade/Support_files/With_catgory_index_Processed_title_des.csv",escape='"',header=True)
train_data = spark.read.csv("With_catgory_index_Processed_title_des.csv", escape='"',
                            schema=data_struc, header=True)
# #only take samples from Fashion category
test_sample = train_data.na.fill({"description": ""})
# title feature extraction
tokenizer = Tokenizer(inputCol="new_title", outputCol="original_words")
regexTokenizer = RegexTokenizer(inputCol="new_title", outputCol="regex_words", pattern="\\W")
stop_words_remover = StopWordsRemover(inputCol="regex_words", outputCol="nsw_words")
# count title character length (including space and punctuation)
test_1 = test_sample.withColumn("title_character_count", charCount(test_sample["title"]))
# count original title word
test_1 = tokenizer.transform(test_1)
test_1 = test_1.withColumn("original_tokens", countTokens(test_1["original_words"]))
# count special character(words) in title
test_1 = regexTokenizer.transform(test_1)
test_1 = test_1.withColumn("regex_tokens", usefulToken(test_1["regex_words"], test_1["original_tokens"]))
# count stop words and informational words in title
test_1 = stop_words_remover.transform(test_1)
test_1 = test_1.withColumn("sw_tokens",
                           usefulToken(test_1["nsw_words"], test_1["original_tokens"]) - test_1["regex_tokens"])
test_1 = test_1.withColumn("nsw_tokens", countTokens(test_1["nsw_words"]))
# calculate description and title relationship (num of informational words in title / num of informational words in des)
regexTokenizer = RegexTokenizer(inputCol="new_des", outputCol="regex_des", pattern="\\W")
stop_words_remover = StopWordsRemover(inputCol="regex_des", outputCol="nsw_des")
test_2 = regexTokenizer.transform(test_1)
test_2 = stop_words_remover.transform(test_2)
test_2 = test_2.withColumn("des_tokens", countTokens(test_2["nsw_des"]))
test_2 = test_2.withColumn("tit_des_ratio", test_2["nsw_tokens"] / test_2["des_tokens"]).na.fill({"tit_des_ratio": 1})
def des_same_func(col1, col2):
    return len(set(col1).intersection(set(col2)))
des_same = udf(des_same_func, IntegerType())
test_2 = test_2.withColumn("des_same", des_same("regex_words", "nsw_des"))
# check unique words in title and count repeated times
test_3 = test_2.withColumn("unique_words", uniqueExtract(test_2["nsw_words"]))
test_3 = test_3.withColumn("repeat_tokens", test_3["nsw_tokens"] - countTokens(test_3["unique_words"]))
# assembling features
assembler = VectorAssembler(
    inputCols=["original_tokens", "des_same", "nsw_tokens", "sw_tokens", "regex_tokens", "tit_des_ratio",
               "repeat_tokens", "title_character_count", "clarity"],
    outputCol="features")
test_4 = assembler.transform(test_3)
test_4 = test_4.select("index", "title", "features", "category_1", test_4["concise"].alias("label"))
# test_3.filter("index==1").select("tokens","sw_tokens", "regex_tokens","r_tit_des","count_repeat").show()
# test_4.select("index","features", test_4["concise"].alias("label")).show()
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
evaluator = BinaryClassificationEvaluator()
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10, seed=42, maxDepth=3)
paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxBins, [32, 16]) \
    .addGrid(gbt.minInfoGain, [0.0, 2.0, 5.0]) \
    .build()
cv_gbt = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid_gbt, evaluator=evaluator, numFolds=3)
# individual
category_type = ["Computers & Laptops", "Fashion", "TV, Audio / Video, Gaming & Wearables", \
                 "Mobiles & Tablets", "Health & Beauty", "Home Appliances", "Watches Sunglasses Jewellery", \
                 "Cameras", "Home & Living"]
for cat_name in category_type:
    data_set = test_4.filter(test_4["category_1"] == cat_name)
    (trainingData, testData) = data_set.randomSplit([0.7, 0.3])
    # Train model.  This also runs the indexers.
    gbt_model = cv_gbt.fit(trainingData)
    # Make predictions.
    gbt_predictions = gbt_model.transform(testData)
    # Select example rows to display.
    # predictions.select("prediction", "label", "features").show(5)
    # Select (prediction, true label) and compute test error
    accuracy_gbt = evaluator.evaluate(gbt_predictions)
    print(cat_name)
    print("Test Error = %g" % (1.0 - accuracy_gbt))
(trainingData, testData) = test_4.randomSplit([0.7, 0.3])
# total
# Train model.  This also runs the indexers.
gbt_model = cv_gbt.fit(trainingData)
# Make predictions.
gbt_predictions = gbt_model.transform(testData)
# Select example rows to display.
# predictions.select("prediction", "label", "features").show(5)
# Select (prediction, true label) and compute test error
accuracy_gbt = evaluator.evaluate(gbt_predictions)
print("Whole dataset")
print("Test Error = %g" % (1.0 - accuracy_gbt))
gbt_predictions.groupBy("label", "prediction").count().show()
