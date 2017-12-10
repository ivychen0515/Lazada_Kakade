from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer
from nltk.stem import SnowballStemmer
import re
import spacy
import csv
import pandas as pd
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors


def remove_char(col):
    if col is not None:
        return [st.stem(w.decode('utf-8')) for w in re.sub(r"[\'\"</&\[\+>()-]", " ", col.encode('utf-8')).split()]

    else:
        return []


def remove_sp_char(col):
    if col is not None:
        return re.sub(r"[\'\"</&\[\+>()-]", " ", col.encode('utf-8'))

    else:
        return col


def cstem(col):
    if col is not None:
        return [st.stem(w) for w in col]

    else:
        return []


# def remove_char2(col):
#     if col is not None:
#         return [st.stem(w.decode('utf-8')) for w in re.sub(r"[\'\"</&\[\+>()-]", " ", col).split()]
#     else:
#         return []
#
# def ent(col):
#     doc = []
#     for w in col.split():
#         d = nlp(w)
#         res = []
#         for token in d:
#             res.append([token.pos_.encode("utf-8"), token.text.encode("utf-8")])
#         doc.append(res)
#     return doc


spark = SparkSession.builder.appName("kakade").getOrCreate()
df = spark.read.csv("Clean_Data_Frame.csv", escape='"', header=True)
st = SnowballStemmer('english')
nlp = spacy.load('en')

df = df.na.fill({"description": ""})

title_clean = udf(remove_char, ArrayType(StringType()))
title_clean_string = udf(remove_char, StringType())
title_remove_sp = udf(remove_sp_char, StringType())
# df = df.withColumn("new_title", title_clean("title"))
df = df.withColumn("title_sp", title_remove_sp("title"))
# df.select("title", "new_title", "title_sp", "concise", "clarity").show(n=100, truncate=False)
# df = df.withColumn("new_des", title_clean("description"))
df = df.withColumn("des_sp", title_remove_sp("description"))

regexTokenizer = RegexTokenizer(inputCol="title_sp", outputCol="regex_title", pattern="\\W")
# regexTokenizer = RegexTokenizer(inputCol="new_title_str", outputCol="regex_title_stem", pattern="\\W")
df = regexTokenizer.transform(df)
# df.select("regex_title_stem").show(10, truncate=False)
df.select("regex_title").show(10, truncate=False)

regexTokenizer2 = RegexTokenizer(inputCol="des_sp", outputCol="regex_des", pattern="\\W")
df = regexTokenizer2.transform(df)
df.select("regex_des").show(10, truncate=False)

stop_words_remover = StopWordsRemover(inputCol="regex_title", outputCol="new_title")
df = stop_words_remover.transform(df)
df.select("new_title").show(5, truncate=False)
# stem
title_clean_2 = udf(cstem, ArrayType(StringType()))
# title_clean_3 = udf(remove_char2, ArrayType(StringType()))
df = df.withColumn("new_title_stem", title_clean_2("new_title"))

stop_words_remover = StopWordsRemover(inputCol="regex_des", outputCol="new_des")
df = stop_words_remover.transform(df)
df.select("new_des").show(5, truncate=False)

# stem
# df = df.withColumn("new_des_stem", title_clean_3("des_sp"))
df = df.withColumn("new_des_stem", title_clean_2("new_des"))
df.select("new_des_stem").show(5, truncate=False)

# df.show(10, truncate=False)

df = df.withColumn("cat_1_stem", title_clean("category_1"))
df.select("cat_1_stem").show(10)

df = df.withColumn("cat_2_stem", title_clean("category_2"))
df.select("cat_2_stem").show(10)

df = df.withColumn("cat_3_stem", title_clean("category_3"))
df.select("cat_3_stem").show(10)

df_full = df.select("index", "title", "description", "title_sp", "des_sp", "concise", "clarity")
df = df.select("index", "new_title_stem", "new_des_stem", "cat_1_stem", "cat_2_stem", "cat_3_stem", "concise", "clarity")
df.show(20, truncate=False)
df_full.show(20, truncate=False)

#############

def count_same(col1, col2, col3, col4):
    total_set = set(col2).union(set(col3)).union(set(col4))
    return len(set(col1).intersection(total_set))


same_word = udf(count_same, IntegerType())
df = df.withColumn("title_cat_same", same_word("new_title_stem", "cat_1_stem", "cat_2_stem", "cat_3_stem"))
df.show(20)

# def title_des(col1, col2):
#     total_set = set(col2)
#     return len(set(col1).intersection(total_set))
#
#
# title_des_same = udf(title_des, IntegerType())
# df = df.withColumn("title_des", same_word("new_title_stem", "new_des_stem"))
# df.show(20)


# df.repartition(1).write.mode("overwrite").json("test")

# df.select("index", "title", "new_title", "description", "new_des", "category_1", "concise", "clarity").repartition(1).write.csv("clean_des", escape='"', header=True)
# df.select("new_des").orderBy("index").repartition(1).write.csv("clean_des", escape='"', header=True)

# title_ent = udf(ent, ArrayType(StringType()))
# df = df.withColumn("title_ent", title_ent("title"))
# df.select("title", "title_ent").show(n=40, truncate=True)


#############
# Extract word entity

mdf = df_full.select("title").rdd.flatMap(lambda x: x).collect()
doc = []
for w in mdf:
    d = nlp(w)
    res = []
    for token in d:
        res.append((token.pos_.encode("utf-8"), token.text.encode("utf-8")))
    doc.append(res)

# for row in doc:
#     for entry in row:
#         entry[0] = entry[0].encode("utf-8")
#         entry[1] = entry[1].encode("utf-8")

# with open("file.csv", 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     for row in doc:
#         wr.writerow(row)

res = []
for row in doc:
    count = 0
    for entry in row:
        if entry[0] == 'NOUN' or entry[0] == 'PROPN':
            count += 1
    ratio = count / float(len(row))
    res.append(ratio)

# with open("noun.csv", 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     for row in res:
#         wr.writerow([row])

pd_df = pd.DataFrame(doc)
pd_df.to_pickle("word_entity.bin")

# ent = []
# for w in mdf:
#     ent.append(nlp(w).ents)

# r = []
# for row in ent:
#     temp = []
#     for entry in row:
#         temp.append([entry.text.encode("utf-8"), entry.label_.encode("utf-8")])
#     r.append(temp)

# with open("column.csv", 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     for row in r:
#         wr.writerow(row)


#############
# LDA

remover = StopWordsRemover(inputCol="new_title_stem", outputCol="new_title_stem_rm",
                           stopWords=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "intl"])
df_lda = remover.transform(df)
df_lda.select("new_title_stem_rm").show(10, truncate=False)
vectorizer = CountVectorizer(inputCol="new_title_stem_rm", outputCol="vec")


########################
# LDA - Extract Category

test = spark.sparkContext.parallelize([(["fashion"]), (["watch, sunglass, jewelleri"])]).toDF(["cat"])
test = test.withColumn("cat_new", split("cat", ", "))
test_1 = df_lda.join(test, df.cat_1_stem == test.cat_new)
test_1 = test_1.drop("cat", "cat_new")
test_1.show()
# test_1.count()
test_2 = df_lda.subtract(test_1)
# test_2.count()


vaf = vectorizer.fit(test_1)
af = vaf.transform(test_1).select("index", "vec")
afr = af.rdd.map(lambda x: [long(x["index"]), Vectors.fromML((x["vec"]))]).cache()
ldam = LDA.train(afr, k=5, optimizer="online")
wordNumbers = 30
topicIndices = ldam.describeTopics(maxTermsPerTopic=wordNumbers)
topicIndices = spark.sparkContext.parallelize(topicIndices)
vocabarray = vaf.vocabulary

vaf2 = vectorizer.fit(test_2)
af2 = vaf2.transform(test_2).select("index", "vec")
afr2 = af2.rdd.map(lambda x: [long(x["index"]), Vectors.fromML((x["vec"]))]).cache()
ldam2 = LDA.train(afr2, k=5, optimizer="online")
wordNumbers2 = 30
topicIndices2 = ldam2.describeTopics(maxTermsPerTopic=wordNumbers2)
topicIndices2 = spark.sparkContext.parallelize(topicIndices2)
vocabarray2 = vaf2.vocabulary


def topic_render(topic):  # specify vector id of words to actual words
    terms = topic[0]
    result = []
    for i in range(wordNumbers):  # top 10
        term = vocabarray[terms[i]]
        result.append(term)
    return result


def topic_render_2(topic):  # specify vector id of words to actual words
    terms = topic[0]
    result = []
    for i in range(wordNumbers2):  # top 10
        term = vocabarray2[terms[i]]
        result.append(term)
    return result

topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()
topics_final2 = topicIndices2.map(lambda topic: topic_render_2(topic)).collect()

for topic in range(len(topics_final)):
    print ("Topic_cat_1_" + str(topic) + ":")
    for term in topics_final[topic]:
        print (term)
    print ('\n')

for topic in range(len(topics_final2)):
    print ("Topic_cat_2_" + str(topic) + ":")
    for term in topics_final2[topic]:
        print (term)
    print ('\n')


def topic_count(col):
    res = []
    ma = 0
    for i in range(0, len(topics_final)):
        num = len(set(col).intersection(set(topics_final[i])))
        if num != 0:
            res.append(num)
            if num > ma:
                ma = num
    if not res:
        return [0, 0]
    return [ma, len(res)]


def topic_count_2(col):
    res = []
    ma = 0
    for i in range(0, len(topics_final2)):
        num = len(set(col).intersection(set(topics_final2[i])))
        if num != 0:
            res.append(num)
            if num > ma:
                ma = num
    if not res:
        return [0, 0]
    return [ma, len(res)]


# def topic_count(col):
#     res = []
#     for i in range(0, len(topics_final)):
#         res.append(set(col).intersection(set(topics_final[i])))
#     return res


tp = udf(topic_count, ArrayType(StringType()))
tp2 = udf(topic_count_2, ArrayType(StringType()))

test_1 = test_1.withColumn("new_feature", tp("new_title_stem_rm"))
test_1.select("index", "new_title_stem", "new_feature", "clarity").filter("clarity==0").show(20, truncate=False)

test_2 = test_2.withColumn("new_feature", tp2("new_title_stem_rm"))
test_2.select("index", "new_title_stem", "new_feature", "clarity").filter("clarity==0").show(20, truncate=False)

# result = set(topics_final[0])
# for s in range(1, len(topics_final)):
#     result = result.intersection(set(topics_final[s]))
#     print result

# test_1 = test_1.drop("cat", "cat_new")
# test_2 = test_2.drop("cat", "cat_new")
df_final = test_1.union(test_2)


def to_int(col):
    return (int(col))


int_udf = udf(to_int, IntegerType())
df_final = df_final.withColumn("index_int", int_udf("index")).sort("index_int")
df_final.show()
