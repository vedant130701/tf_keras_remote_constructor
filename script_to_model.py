import tensorflow as tf
import numpy as np
import os
import pandas as pd
from contextlib import redirect_stdout

model_name = ""

def read_script(path):
	file = open(path, 'r')
	model_description_full = file.readlines()
	model_description_full = [i[:-1].strip() for i in model_description_full]  #removing '\n'
	file.close()
	return model_description_full

def script_to_model(model_description_full):
	model_evals = {
	'epochs': int(model_description_full[0][len("epochs: "):]),
	'loss': model_description_full[1][len("loss: "):],
	'optimizer': model_description_full[2][len("optimizer: "):],
	'metrics': model_description_full[3][len("metrics: ["):-1].split(', ')
	}

	model_description = [i.split(', ') for i in model_description_full[model_description_full.index('model:')+1:]]
	
	return model_evals, model_description

class myCallback(tf.keras.callbacks.Callback):
	global model_name
	def on_epoch_end(self, epoch, logs={}): 
		current_evals = {"epoch": [epoch]}
		for key, item in logs.items():
			current_evals.update({key: [item]})
		current_evals = pd.DataFrame(current_evals)
		if epoch == 1:
			current_evals.to_csv('./output_'+model_name+'/per_epoch.csv', index=False)
		else:
			current_evals.to_csv('./output_'+model_name+'/per_epoch.csv', header=False, mode='a', index=False)

		

def create_model(model_evals, model_description):
	model_arr = []
	for layer in model_description:
		if layer[0] == 'Dense':
			model_arr.append(tf.keras.layers.Dense(int(layer[1]), activation=layer[2]))
		elif layer[0] == 'Dropout':
			model_arr.append(tf.keras.layers.Dropout(float(layer[1])))

	model = tf.keras.Sequential(model_arr)

	model.compile(loss=model_evals["loss"],
                optimizer=model_evals["optimizer"],
                metrics=model_evals["metrics"])

	return model

def data_reader(data_path):
	X = pd.read_csv(data_path+'X.csv')
	Y = pd.read_csv(data_path+'Y.csv')
	X = X.values
	Y = Y.values

	x_train = X[:int(X.shape[0]*0.8)]
	x_test = X[int(X.shape[0]*0.8):]
	y_train = Y[:int(X.shape[0]*0.8)]
	y_test = Y[int(X.shape[0]*0.8):]

	return x_train, y_train, x_test, y_test

def main_script_to_model(path, name, save_history=False):
	global model_name
	model_name = name

	os.system('mkdir output_'+model_name)

	data_path = './data/'

	x_train, y_train, x_test, y_test = data_reader(data_path)

	model_description_full = read_script(path)

	model_evals, model_description = script_to_model(model_description_full)

	model = create_model(model_evals, model_description)

	callbacks = myCallback()

	if save_history:
		with open('./output_'+model_name+'/history.txt', 'w+') as f:
			with redirect_stdout(f):
				history = model.fit(
					x_train, y_train,
					epochs=model_evals["epochs"],
					validation_data=(x_test, y_test),
					callbacks=[callbacks],
					verbose=1
					)
	else:
		history = model.fit(
			x_train, y_train,
			epochs=model_evals["epochs"],
			validation_data=(x_test, y_test),
			callbacks=[callbacks],
			verbose=1
			)


	

	evals_train_raw = model.evaluate(x_train, y_train)
	evals_test_raw = model.evaluate(x_test, y_test)

	labels = ["loss"] + model_evals["metrics"]

	evals_train = {}
	evals_test = {}

	for index, label in enumerate(labels):
		evals_train.update({label: evals_train_raw[index]})
		evals_test.update({label: evals_test_raw[index]})

	final_value = {"label": ["train", "test"]}

	for label in labels:
		final_value.update({label: [evals_train[label], evals_test[label]]})

	final_value = pd.DataFrame(final_value)

	final_value.to_csv('./output_'+model_name+'/final_output.csv', index=False)


	with open('./output_'+model_name+'/summary.txt', 'w+') as f:
		with redirect_stdout(f):
			model.summary()



if __name__ == "__main__":
	os.system('clear')
	main_script_to_model('./test/example_params.txt', 'simple_logistic')
