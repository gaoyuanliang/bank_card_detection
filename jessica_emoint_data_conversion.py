#########jessica_emoint_data_conversion.py#########

import re
import csv
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

sc = SparkContext("local")
sqlContext = SparkSession.builder.getOrCreate()

def str2float(input):
	try:
		return float(input)
	except:
		return None

#str2float("7.567")

'''
load the files to data 
'''

'''
emotion_tag = "fear"
data_file = "*.train.txt"

texts, tags = convert_file_to_text_and_tag_list(
	emotion_tag,
	data_file)
'''

def convert_file_to_text_and_tag_list(
	emotion_tag,
	data_file):
	udf_str2float = udf(str2float, FloatType())
	schema = StructType()\
		.add("tweet_id",StringType(),True)\
		.add("text",StringType(),True)\
		.add("emotion",StringType(),True)\
		.add("intensity",FloatType(),True)
	train = sqlContext.read.format('csv')\
		.options(delimiter='\t')\
		.schema(schema)\
		.load(data_file)\
		.withColumn("intensity", udf_str2float("intensity"))
	train.registerTempTable("train")
	train_list = sqlContext.sql(u"""
		SELECT *, 
		CASE 
			WHEN emotion = '%s' THEN 1
			ELSE 0
		END AS label
		FROM train
		WHERE emotion IS NOT NULL
		"""%(emotion_tag)).collect()
	texts = [r.text for r in train_list]
	tags = [r.label for r in train_list]
	return texts, tags

def convert_file_to_text_and_score_list(
	emotion_tag,
	data_file):
	udf_str2float = udf(str2float, FloatType())
	schema = StructType()\
		.add("tweet_id",StringType(),True)\
		.add("text",StringType(),True)\
		.add("emotion",StringType(),True)\
		.add("intensity",FloatType(),True)
	train = sqlContext.read.format('csv')\
		.options(delimiter='\t')\
		.schema(schema)\
		.load(data_file)\
		.withColumn("intensity", udf_str2float("intensity"))
	train.registerTempTable("train")
	train_list = sqlContext.sql(u"""
		SELECT *
		FROM train
		WHERE emotion = '%s' AND intensity IS NOT NULL
		"""%(emotion_tag)).collect()
	texts = [r.text for r in train_list]
	scores = [r.intensity for r in train_list]
	return texts, scores

#########jessica_emoint_data_conversion.py#########