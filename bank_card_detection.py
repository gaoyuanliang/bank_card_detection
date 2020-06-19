########bank_card_detection.py########
from pavi import *

model = load_build_image_categorization_model(
	model_file = 'bank_card.h5py')

def bank_card_detection(input_file):
	output = {}
	x = read_image_from_local(input_file)
	x = xception.preprocess_input(x)
	x = numpy.array([x])
	x = base_model_Xception.predict(x)
	y_score = model.predict(x)
	prediction = numpy.argmax(y_score)
	score = numpy.max(y_score)
	if prediction > 0:
		output["tag"] = 'bank_card'
		output["score"] = score
	return output
########bank_card_detection.py########
