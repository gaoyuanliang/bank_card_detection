import time
import numpy
import hashlib
from PIL import *
from keras.utils import *
from keras.losses import *
from keras.layers import *
from keras.metrics import *

from jessica_local_spark_building import sqlContext
from pyspark.sql.types import StructType, StructField, StringType

from pyspark import StorageLevel

from keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception

base_model_Xception = xception.Xception(weights='xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

def file_json2file_npy(input_json,
	sqlContext,
	file_path_column_name = None,
	x_xception_npy = None,
	x_document_id_npy = None,
	y_npy = None,
	output_json = None):
	start_time = time.time()
	print('loading data from %s'%(input_json))
	input_df = sqlContext.read.json(input_json)
	input_df.registerTempTable('input_df')
	input_df = sqlContext.sql(u"""
		SELECT * FROM input_df ORDER BY document_id
		""")
	print('loaded %d records from %s'%(input_df.count(), input_json))
	print('collecting data')
	data  =input_df.collect()
	###
	x_document_id = []
	x_file_path = []
	x_xception = []
	x_inception_v3 = []
	x_hash = []
	y = []
	for r in data:
		r1 = r.asDict()
		if x_xception_npy is not None:
			img = image.load_img(r1[file_path_column_name], target_size=(224, 224))
			x = image.img_to_array(img)
			if output_json is not None:
				hash_object = hashlib.md5(str(x.data.tobytes()).encode())
				x_hash.append(hash_object.hexdigest())
			x_xception.append(xception.preprocess_input(x))
			x_document_id.append(r.document_id)
		if y_npy is not None:
			y.append(r.label)
	if x_xception_npy is not None:
		print('extracting and saving featurs')
		x_document_id = numpy.array(x_document_id)
		numpy.save(x_document_id_npy, x_document_id)
		x_xception = numpy.array(x_xception)
		x_xception = base_model_Xception.predict(x_xception)
		numpy.save(x_xception_npy, x_xception)
	if y_npy is not None:
		print('saving labels')
		y = numpy.array(y)
		y = to_categorical(y)
		numpy.save(y_npy, y)
	if output_json is not None:
		if 'content_hash' not in input_df.columns:
			data = [(str(d), str(h)) 
				for d, h in 
				zip(x_document_id, x_hash)]
			sqlContext.createDataFrame(data, ['document_id', 'content_hash']).registerTempTable('df_content_hash')
			sqlContext.sql(u"""
				SELECT input_df.*,
				df_content_hash.content_hash
				FROM input_df
				LEFT JOIN df_content_hash
				ON df_content_hash.document_id
				= input_df.document_id
				""").write.mode('Overwrite').json(output_json)
		else:
			input_df.write.mode('Overwrite').json(output_json)
	print('running time:\t%f secondes'%(time.time()-start_time))
	return None

'''
>>> x_xception.shape
(8, 7, 7, 2048)
>>> x_inception_v3.shape
(8, 5, 5, 2048)
'''

def build_image_categorization_model(gpus = None):
	model = Sequential()
	model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	if gpus is not None:
		model = multi_gpu_model(model, gpus = gpus)
	return model

def train_image_categorization_model(
	x_npy, y_npy,
	x_document_id_npy,
	gpus = None,
	epochs = 3,
	positive_weight = 1,
	batch_size = 512,
	model_file = None,
	output_prediction_json = None):
	#####
	print('load data and label from npy files')
	x = numpy.load(x_npy)
	x_document_id = numpy.load(x_document_id_npy)
	y = numpy.load(y_npy)
	####
	print('building model')
	model = build_image_categorization_model(gpus = gpus)
	model.compile(loss='categorical_crossentropy',
		optimizer='rmsprop', 
		metrics=['accuracy'])
	print('training the model')
	model.fit(x, y, 
		batch_size=batch_size, 
		epochs=epochs,
		class_weight = {1:positive_weight, 0:1})
	if model_file is not None:
		print('saving the model')
		model.save_weights(model_file)
	#####
	print('predicting the labels from the trained model')
	y_score = model.predict(x)
	label_predicted = numpy.argmax(y_score,axis=-1)
	label = numpy.argmax(y,axis=-1)
	label_confidence = numpy.max(y_score,axis=1)
	print('building the dataframe of the prediciton results')
	data = [(str(d), int(l), int(p), float(s)) 
		for d, l, p, s in zip(x_document_id,
		label,
		label_predicted,
		label_confidence)]
	###
	df_prediction = sqlContext.createDataFrame(data, 
	['document_id', 'label', 'prediction', 'score']).persist(StorageLevel.MEMORY_AND_DISK)
	####
	if output_prediction_json is not None:
		print('saving the prediction results')
		df_prediction.write.mode('Overwrite').json(output_prediction_json)
	#####
	df_prediction.registerTempTable('df_prediction')
	sqlContext.sql(u"""
		SELECT label, prediction, COUNT(*)
		FROM df_prediction
		GROUP BY label, prediction
		""").show()
	return model

def update_label_from_positive_file_csv(
	input_json,
	output_json,
	sqlContext,
	input_positive_file_csv = None,
	input_positive_hash_csv = None):
	####
	'''
	input_positive_file_csv = 'uae_flag.csv'
	input_json = 'image_set.json'
	output_json = 'uae_flag_updated_label.json'
	'''
	sqlContext.read.json(input_json).withColumnRenamed('label', 'old_label').registerTempTable('input_df')
	####
	if input_positive_file_csv is not None:
		additioal_postive_file = sqlContext.read.format('csv')\
		.option("header", "false")\
		.schema(StructType([StructField("document_id", StringType(), True)]))\
		.load(input_positive_file_csv)\
		.dropDuplicates()
		additioal_postive_file.registerTempTable('additioal_postive_file')
		print('loaded %d positive document_id'%(additioal_postive_file.count()))
	else:
		sqlContext.sql(u"""
			SELECT NULL AS document_id
			""").registerTempTable('additioal_postive_file')
	####
	if input_positive_hash_csv is not None:
		additioal_postive_hash = sqlContext.read.format('csv')\
		.option("header", "false")\
		.schema(StructType([StructField("content_hash", StringType(), True)]))\
		.load(input_positive_hash_csv)\
		.dropDuplicates()
		additioal_postive_hash.registerTempTable('additioal_postive_hash')
		print('loaded %d positive content hash'%(additioal_postive_hash.count()))
	else:
		sqlContext.sql(u"""
			SELECT NULL AS content_hash
			""").registerTempTable('additioal_postive_hash')
	###
	sqlContext.sql(u"""
		SELECT input_df.*,
		CASE 
			WHEN additioal_postive_file.document_id IS NOT NULL  
			OR additioal_postive_hash.content_hash IS NOT NULL
			THEN 1
			ELSE old_label 
		END AS label
		FROM input_df
		LEFT JOIN additioal_postive_file
		ON  additioal_postive_file.document_id
		= input_df.document_id
		LEFT JOIN additioal_postive_hash
		ON  additioal_postive_hash.content_hash
		= input_df.content_hash
		ORDER BY input_df.document_id
		""").write.mode('Overwrite').json(output_json)
	sqlContext.read.json(output_json).registerTempTable('output_df')
	sqlContext.sql(u"""
		SELECT old_label, label, COUNT(*)
		FROM output_df
		GROUP BY old_label, label
		""").show()

def load_build_image_categorization_model(
	model_file,
	gpus = None):
	model = build_image_categorization_model(gpus = gpus)
	model.load_weights(model_file)
	model.compile(loss='categorical_crossentropy',
		optimizer='rmsprop', 
		metrics=['accuracy'])
	model._make_predict_function()
	return model

def image_tagging(
	x, model,
	tag_name):
	output = {}
	x = xception.preprocess_input(x)
	x = numpy.array([x])
	x = base_model_Xception.predict(x)
	y_score = model.predict(x)
	prediction = numpy.argmax(y_score)
	score = numpy.max(y_score)
	if prediction > 0:
		output["tag"] = tag_name
		output["score"] = score
	return output

def read_image_from_local(file_path):
	img = image.load_img(file_path, target_size=(224, 224))
	x = image.img_to_array(img)
	return  x
