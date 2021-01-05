##########jessica_deep_emotion_sensor.py##########
import numpy as np
import keras
from keras import *
from keras.utils import *

# Model constants.
max_features = 20000
embedding_dim = 300
sequence_length = 100


'''
def text_processing(t):
	t = t.lower()
	t = re.sub(r'\s+', r' ', t)
	t = r' '+t+r' '
	for m in re.finditer(r'[^a-z][a-z]+[^a-z]', t):
		original = m.group()
		word = re.search(r'[a-z]+', original).group()
		replaced = re.sub(word, r' '+word+r' ', original)
		t = re.sub(re.escape(original), replaced, t)
	for m in re.finditer(r'[^a-z][a-z]+[^a-z]', t):
		original = m.group()
		word = re.search(r'[a-z]+', original).group()
		replaced = re.sub(word, r' '+word+r' ', original)
		t = re.sub(re.escape(original), replaced, t)
	t = t.strip()
	t = re.sub(r'\s+', r' ', t)
	return t

	texts = [text_processing(t) for t in texts]
'''

'''
print(text_processing(u"Don't join @BTCare they put the phone down on you, talk over you and are rude. Taking money out of my acc willynilly! #fuming"))
'''

def texts_to_input(texts):
	word_id_sequence = map(lambda x: keras.preprocessing.text.one_hot(x, n=max_features), 
		texts)
	word_id_sequence = list(word_id_sequence)
	x = np.array(word_id_sequence)
	x = keras.preprocessing.sequence.pad_sequences(
		x, padding="post",
		maxlen=sequence_length,
	)
	return x

def emotion_tagger_model_building(
	max_features = 20000,
	embedding_dim = 300,
	sequence_length = 100,
	dropout_rate = 0.2):
	# A integer input for vocab indices.
	inputs = keras.Input(shape=(sequence_length,), dtype="int64")
	# Next, we add a layer to map those vocab indices into a space of dimensionality
	# 'embedding_dim'.
	x = layers.Embedding(max_features, embedding_dim)(inputs)
	x = layers.Dropout(dropout_rate)(x)
	# Conv1D + global max pooling
	x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
	x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
	x = layers.GlobalMaxPooling1D()(x)
	# We add a vanilla hidden layer:
	x = layers.Dense(128, activation="relu")(x)
	x = layers.Dropout(dropout_rate)(x)
	# We project onto a single unit output layer, and squash it with a sigmoid:
	predictions = layers.Dense(2, 
		activation="softmax",
		 name="predictions")(x)
	model = keras.Model(inputs, predictions)
	model.compile(
		loss="categorical_crossentropy", 
		optimizer="adam", 
		metrics=["accuracy"])
	return model

def train_tagger(texts,
	tags,
	tagger_model_path = None,
	tagger_model_weight_path = None,
	tagger_model_json_path = None,
	epochs = 100,
	validation_split=0.1,
	dropout_rate = 0.2,
	):
	tagger_model = emotion_tagger_model_building(
		dropout_rate = dropout_rate,
		)
	'''
	prepare the text input

	texts = [
		"i feel so fear",
		"nothing is wrong"
		]
	'''
	x = texts_to_input(texts)
	'''
	prepare the output
	'''
	y = np.array(tags)
	y = to_categorical(y)
	print(x.shape, y.shape)
	print(np.sum(y, axis = 0))
	# Fit the model using the train and test datasets.
	tagger_model.fit(x, y, 
		validation_split=validation_split, 
		epochs=epochs)
	# serialize model to JSON
	if tagger_model_json_path is not None:
		model_json = tagger_model.to_json()
		with open(tagger_model_json_path, 'w+') as json_file:
			json_file.write(model_json)
	# serialize weights to HDF5
	if tagger_model_json_path is not None:
		tagger_model.save_weights(tagger_model_weight_path)
	if tagger_model_path is not None:\
		tagger_model.save(tagger_model_path)
	return tagger_model

'''
# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''

def emotion_scorer_model_building(
	max_features = 20000,
	embedding_dim = 300,
	sequence_length = 100):
	# A integer input for vocab indices.
	inputs = keras.Input(shape=(sequence_length), dtype="int64")
	# Next, we add a layer to map those vocab indices into a space of dimensionality
	# 'embedding_dim'.
	x = layers.Embedding(max_features, embedding_dim)(inputs)
	x = layers.Dropout(0.5)(x)
	# Conv1D + global max pooling
	x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
	x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
	x = layers.GlobalMaxPooling1D()(x)
	# We add a vanilla hidden layer:
	x = layers.Dense(128, activation="relu")(x)
	x = layers.Dropout(0.5)(x)
	# We project onto a single unit output layer, and squash it with a sigmoid:
	predictions = layers.Dense(1, 
		activation="sigmoid",
		 name="predictions")(x)
	model = keras.Model(inputs, predictions)
	model.compile(
		loss="mse", 
		optimizer="adam", 
		metrics=[metrics.mean_absolute_error])
	return model


def train_scorer(texts,
	scores,
	scorer_model_path,
	epochs = 100,
	validation_split=0.1,
	):
	scorer_model = emotion_scorer_model_building()
	'''
	prepare the text input

	texts = [
		"i feel so fear",
		"nothing is wrong"
		]
	'''
	x = texts_to_input(texts)
	'''
	prepare the output
	'''
	y = np.array(scores)
	print(x.shape, y.shape)
	# Fit the model using the train and test datasets.
	scorer_model.fit(x, y, 
		validation_split=0.1, 
		epochs=epochs)
	scorer_model.save(scorer_model_path,
		save_format='h5')
	return scorer_model


##########jessica_deep_emotion_sensor.py##########