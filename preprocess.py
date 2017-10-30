import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructField, StringType, FloatType, StructType
from pyspark.sql.functions import *


# def count_not_null(c, nan_as_null=False):
#     """Use conversion between boolean and integer
#     - False -> 0
#     - True ->  1
#     """
#     pred = col(c).isNotNull() & (~isnan(c) if nan_as_null else lit(True))
#     return sum(pred.cast("integer")).alias(c)


# Add header to data feature
feature_schema = [StructField("country", StringType(), True), StructField("sku_id", StringType(), True),
                  StructField("title", StringType(), True), StructField("category_1", StringType(), True),
                  StructField("category_2", StringType(), True), StructField("category_3", StringType(), True),
                  StructField("description", StringType(), True), StructField("org_price", FloatType(), True),
                  StructField("product_type", StringType(), True)]
feature_struct = StructType(fields=feature_schema)

spark = SparkSession.builder.appName("kakade").getOrCreate()

# country sku_id title category_1 category_2 category_3 short_description price product_type
df = spark.read.csv("Data/training/data_train.csv", escape='"', schema=feature_struct).withColumn("id", monotonically_increasing_id()+1)
w = Window().partitionBy().orderBy(col("id"))  # No partition
df = df.withColumn("index", row_number().over(w))  # escape quote in quote

concise = spark.read.csv("Data/training/conciseness_train.labels").toDF("concise").withColumn("index", monotonically_increasing_id()+1)
clarity = spark.read.csv("Data/training/clarity_train.labels").toDF("clarity").withColumn("index", monotonically_increasing_id()+1)
label = concise.join(clarity, "index")
df = df.select("index", "title", "category_1", "category_2", "category_3", "description", "org_price", "product_type", "country")

# df.show(20)
# w2 = Window().partitionBy("catagory_1").orderBy("index")
df = df.join(label, 'index')

# # check null
# df.agg(*[count_not_null(c) for c in df.columns]).show()
# label.agg(*[count_not_null(c) for c in label.columns]).show()
# df.agg(*[count_not_null(c) for c in df.columns]).show()

df.orderBy("index").show(200)

# count categories
df.groupBy("category_1").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("category_2").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("category_3").count().orderBy("count", ascending=False).show(truncate=False)

# most other category only contain <=5 products
# filter products out of the main categories
# main_category = df.groupBy("product_type").count().filter("count>5")
# main_category.show(30)

# train_feature = train_feature.join(main_category, train_feature.category_1 == main_category.category_1, 'inner')

# from bs4 import BeautifulSoup
# from pyspark.sql.functions import udf
# from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer
#
#
# def html_extract_func(col):
#     soup = BeautifulSoup(col)
#     return soup.get_text()
#     return (1 - col1 / col2) ** 2
#
#
# html_extract = udf(html_extract_func, StringType())
# train_feature = train_feature.withColumn("html_extract", html_extract("description"))
# train_feature.select("html_extract").show(truncate=False)
#
# # regexTokenizer = RegexTokenizer(inputCol="html_extract", outputCol="description_token", pattern="\\W")
# # train_feature = regexTokenizer.transform(train_feature)
# # stop_words_remover = StopWordsRemover(inputCol="description_token", outputCol="stop_words_filtered")
# # train_feature = stop_words_remover.transform(train_feature)
# train_feature.select("stop_words_filtered").show(truncate=False)
#
# # currency_exchange
# # set all price to PHP
# import pyspark.sql.functions as F
# from forex_python.converter import CurrencyRates
#
# cex = CurrencyRates() S2P = cex.get_rate("SGD", "PHP") M2P = cex.get_rate("MYR", "PHP") train_feature =
# train_feature.withColumn("price", when(train_feature.country == "my", M2P * train_feature.org_price).when(train_feature.country == "sg", S2P * train_feature.org_price).otherwise(train_feature.org_price))
#
# train_feature.select("price", "org_price").show()
