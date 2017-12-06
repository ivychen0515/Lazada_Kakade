
# coding: utf-8
#used features:
# "original_tokens" : 原title词数, 
#"nsw_tokens": 非特殊字符、stopwords词数,
#"sw_tokens": stopwords词数, 
#"regex_tokens": 特殊字符词数,
#"tit_des_ratio": 去除特殊字符和stopwords后title与des词数比,
#"repeat_tokens": 重复词数（重复过的词数，不包含第一次）,
#"title_character_count": 字符长度,
#"clarity"




# In[1]:


import findspark
findspark.init('/home/kakade/spark')
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql.functions import udf,regexp_replace
from pyspark.ml.feature import Tokenizer, RegexTokenizer,StopWordsRemover,CountVectorizer,VectorAssembler
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression,DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

spark = SparkSession.builder.appName("kakade").getOrCreate()


# In[2]:


# function define#####

#count number of tokens in given column
countTokens = udf(lambda words: len(words), IntegerType())

#count total character length in title/description
charCount = udf(lambda x: len(x),IntegerType())

#count number of words w/o special/stop words in title/description
def useful_token_func(col1,col2):
    if len(col1)>col2:
        return 0
    else:
        return col2-len(col1)
usefulToken = udf(useful_token_func, IntegerType())

#extract unique words from title
uniqueExtract = udf(lambda words: list(set(words)),ArrayType(StringType()))

#label string to int
string2int = udf(lambda x: int(x),IntegerType())


# In[3]:


#Add header to data feature
data_schema = [StructField("index", IntegerType(), True), StructField("title", StringType(), True),                  StructField("new_title",StringType(), True), StructField("description",StringType(), True),                   StructField("new_des",StringType(), True), StructField("category_1",StringType(), True),                   StructField("concise",IntegerType(), True), StructField("clarity", IntegerType(), True)]
# country sku_id title category_1 category_2 category_3 short_description price product_type 
data_struc = StructType(fields=data_schema)


# In[4]:


#load all data
# train_data=spark.read.csv("/home/kakade/Support_files/With_catgory_index_Processed_title_des.csv",escape='"',header=True)
train_data=spark.read.csv("/home/kakade/Support_files/With_catgory_index_Processed_title_des.csv",escape='"',schema=data_struc)


# In[5]:


#only take samples from Fashion category
test_sample=train_data.filter(train_data["category_1"]=="Fashion").drop("category_1")
test_sample=test_sample.na.fill({"description":""})


# In[6]:


#title feature extraction

tokenizer = Tokenizer(inputCol="new_title", outputCol="original_words")
regexTokenizer = RegexTokenizer(inputCol="new_title", outputCol="regex_words", pattern="\\W")
stop_words_remover = StopWordsRemover(inputCol="regex_words", outputCol="nsw_words")


#count title character length (including space and punctuation)
test_1=test_sample.withColumn("title_character_count",charCount(test_sample["title"]))

#count original title word 
test_1 = tokenizer.transform(test_1)
test_1=test_1.withColumn("original_tokens", countTokens(test_1["original_words"]))

#count special character(words) in title 
test_1 = regexTokenizer.transform(test_1)
test_1=test_1.withColumn("regex_tokens", usefulToken(test_1["regex_words"],test_1["original_tokens"]))

#count stop words and informational words in title 
test_1 = stop_words_remover.transform(test_1)
test_1=test_1.withColumn("sw_tokens", usefulToken(test_1["nsw_words"],test_1["original_tokens"])-test_1["regex_tokens"])
test_1=test_1.withColumn("nsw_tokens", countTokens(test_1["nsw_words"]))


# In[7]:


#calculate description and title relationship (num of informational words in title / num of informational words in des)
regexTokenizer = RegexTokenizer(inputCol="new_des", outputCol="regex_des", pattern="\\W")
stop_words_remover = StopWordsRemover(inputCol="regex_des", outputCol="nsw_des")

test_2 = regexTokenizer.transform(test_1)
test_2 = stop_words_remover.transform(test_2)
test_2=test_2.withColumn("des_tokens", countTokens(test_2["nsw_des"]))
test_2=test_2.withColumn("tit_des_ratio",test_2["nsw_tokens"]/test_2["des_tokens"]).na.fill({"tit_des_ratio":1})


# In[8]:


#check unique words in title and count repeated times
test_3=test_2.withColumn("unique_words",uniqueExtract(test_2["nsw_words"]))
test_3=test_3.withColumn("repeat_tokens",test_3["nsw_tokens"]-countTokens(test_3["unique_words"]))


# In[9]:


#assembling features

assembler = VectorAssembler(
    inputCols=["original_tokens", "nsw_tokens","sw_tokens", "regex_tokens","tit_des_ratio","repeat_tokens","title_character_count","clarity"],
    outputCol="features")

test_4 = assembler.transform(test_3)
test_4=test_4.select("index","title","features", test_4["concise"].alias("label"))

# test_3.filter("index==1").select("tokens","sw_tokens", "regex_tokens","r_tit_des","count_repeat").show()
# test_4.select("index","features", test_4["concise"].alias("label")).show()


# In[10]:


#Train_test_split
(trainingData, testData) = test_4.randomSplit([0.7, 0.3])

#cross_validation_train setup
evaluator = BinaryClassificationEvaluator()
#LR model
lr = LogisticRegression(maxIter=10)

paramGrid_lr = ParamGridBuilder()    .addGrid(lr.regParam, [0.1, 0.01])     .addGrid(lr.fitIntercept, [False, True])    .build()

cv_lr = CrossValidator(estimator=lr,estimatorParamMaps=paramGrid_lr,evaluator=evaluator,numFolds=3)


#Decision Tree model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

paramGrid_dt = ParamGridBuilder()    .addGrid(dt.maxDepth, [2, 3, 5]).build()
    
cv_dt = CrossValidator(estimator=dt,estimatorParamMaps=paramGrid_dt,evaluator=evaluator,numFolds=3)    
# Run Crossvalidation, and choose the best set of parameters.
lr_model = cv_lr.fit(trainingData)
dt_model = cv_dt.fit(trainingData)


# In[11]:


# Make predictions.

lr_prediction = lr_model.transform(testData)
dt_prediction = dt_model.transform(testData)

accuracy_lr = evaluator.evaluate(lr_prediction)
accuracy_dt = evaluator.evaluate(dt_prediction)
print("LR accuracy = %g " % accuracy_lr)
print("DT accuracy = %g " % accuracy_dt)

