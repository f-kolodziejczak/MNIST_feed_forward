import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math
import numpy as np



# Wyświetlanie wielu przykładów po kolei
def show_labelled_examples(net_input, net_output, category_names, example_count = 10):
	# Wyliczamy ile wierszy i kolumn musi mieć siatka żeby zmieścić wszystkie przykłady
	# a następnie tworzymy okienko pyplota z taką siatką
	grid_size = int(math.ceil(math.sqrt(example_count)))
	plt.figure(figsize=(10, 10))

	#Generowanie obrazu dla każdego przykładu
	for i in range(example_count):
		plt.subplot(grid_size, grid_size, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)

		index = random.randint(0, len(net_input))

		plt.imshow(net_input[index], cmap=plt.cm.binary)
		category_name = category_names[net_output[index]]
		plt.xlabel(category_name)

	plt.waitforbuttonpress()
	plt.close()



# Metoda ucząca sieć
def teach(network, net_input, net_labels, epochs=5):
	# Proces uczenia sieci
	network.fit(net_input,net_labels, epochs=epochs)


# Testowanie naszej sieci na innych przykładach
def evaluate_network(network, net_input,net_labels):
	loss, accuracy = network.evaluate(net_input, net_labels)
	print('Network accuracy:', accuracy)
	print('Network loss:', loss)

# Metoda dokonująca pojedyńcych klasyfikiacji, wizualizując wynik w postaci wykresów słupkowych
def random_classification(network, net_input,net_labels, categories, count=1):
	for i in range(count):

		img = random.choice(net_input)

		#Zamiana obrazu na wsad (lista obrazów)
		batch = np.expand_dims(img,0)
		predictions = network.predict(batch)


		# GUI
		plt.title("Przykładowe klasyfikacje")
		plt.figure(figsize=(12,12))
		plt.subplot(2,1,1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(img)
		plt.xlabel(categories[np.argmax(predictions)])

		plt.subplot(2,1,2)
		plt.grid(False)
		plt.xticks([])
		plt.yticks([])
		# rozkład słupkowy - na ile liczba należy do danej kategorii
		plt.bar(range(len(categories)), predictions[0], color = "#777777")    		
		plt.ylim([0,1])
		plt.xticks(range(len(categories)), categories, rotation=45)

		plt.waitforbuttonpress()
		plt.close()


def build_network(net_input, net_labels, categories):

    # Zmiana wejścia w jednowymiarowy wektor.
    flattener = keras.layers.Flatten(input_shape=(int(format(net_input.shape[1])),
                                                  int(format(net_input.shape[2]))))		    	


    # Wartstwa wejściowa, z funkcją aktywacyjną "relu".
    input_layer = keras.layers.Dense(128, activation=tf.nn.relu)


    # Warstwa wyjściowa, z funkcją aktywacji softmax - normalizuje wektor nadając im 
    # wartości między 0 a 1.
    output_layer = keras.layers.Dense(len(categories), activation=tf.nn.softmax)


    # Tworzenie modelu sieci neuronowej typu feed forward
    model = keras.Sequential([flattener, input_layer, output_layer])

    # Kompilacja modelu.
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )

    return model


# Funkcja wczytująca dataset z kerasa.
def load_dataset():

	(train_images,train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

	# Normalizujemy wejścia. W zbiorze danych każdy piksel ma wartość od 0 do 255 podczas gdy
	# na wejściu sieci potrzebujemy wartości od 0 do 1
	train_images = train_images / 255.0
	test_images = test_images / 255.0

	# Tworzymy obiekty
	input_train = train_images
	output_train = train_labels
	input_test = test_images
	output_test = test_labels

	# Nazwy kategorii w podanym zbiorze danych
	category_names = ['0','1','2','3','4','5','6','7','8','9']

	return input_train, output_train, input_test, output_test, category_names

def main():
	training_input,training_labels,test_input,test_labels, category_names = load_dataset()
	show_labelled_examples(training_input,training_labels,category_names, 10)
	net = build_network(training_input, training_labels, category_names)
	print("Przeprowadzam uczenie sieci")
	teach(net,training_input, training_labels)
	print("Testowanie jakości sieci")
	evaluate_network(net, test_input, test_labels)
	print('Przykładowe klasyfikacje')
	random_classification(net, test_input, test_labels, category_names, 5)


if __name__ == '__main__':
	main()











