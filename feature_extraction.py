import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer
from nltk.stem import SnowballStemmer
import re
import spacy
import pandas as pd


def remove_char(col):
    return [st.stem(w.decode('utf-8')) for w in re.sub(r"[\'\"</\[\+>()-]", " ", col.encode('utf-8')).split()]


# def ent(col):
#     doc = []
#     for w in nlp(col.decode('utf-8')).ents:
#         doc.append(str(w))
#     return doc

spark = SparkSession.builder.appName("kakade").getOrCreate()
df = spark.read.csv("Clean_Data_Frame.csv", escape='"', header=True)
st = SnowballStemmer('english')
nlp = spacy.load('en')

import spacy_func
spark.Sparkcontext.addPyFile('spacy_func.py')

title_clean = udf(remove_char, StringType())
df = df.withColumn("new_title", title_clean("title"))
df.select("title", "new_title").show(n=40, truncate=False)

title_ent = udf(spacy_func.spacy_func, ArrayType(StringType()))
# df = df.withColumn("title_ent", title_ent("title"))
# df.select("title", "title_ent").show(n=40, truncate=True)

