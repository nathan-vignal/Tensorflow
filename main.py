# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)


nomHabit = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
habitMnist = keras.datasets.fashion_mnist

(imageEntrainement, labelEntrainement), (imageTest, labelTest) = habitMnist.load_data()
# output array is read only
imageEntrainement = imageEntrainement / 255
imageTest = imageTest / 255

'''
plt.figure(figsize=(100,100))
for i in range (0,100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imageEntrainement[i])
plt.show()'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(imageEntrainement, labelEntrainement, epochs=5, batch_size=20)

test_loss, test_acc = model.evaluate(imageTest,
                                     labelTest)  # evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
# avec x test data et y test labels

print('Test accuracy:', test_acc)
predictions = model.predict(imageTest)
print(predictions[0])
print(labelTest[0])
'''
def plot_image(i, predictions_array, true_label, img):
    predictions_array = predictions_array[i]
    true_label = true_label[i]
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)'''






