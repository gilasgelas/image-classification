import os
import numpy
import matplotlib.pyplot as plt
import pickle
import keras
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
import PIL

K.common.set_image_dim_ordering('tf')
seed = 7
numpy.random.seed(seed)

# Prepare training data
dataset_dir="dataset"
label = os.listdir(dataset_dir)
save_label = open("labels.txt","wb")
pickle.dump(label, save_label)
save_label.close()
dataset=[]
for image_label in label:
    images = os.listdir(dataset_dir+"/"+image_label)
    for image in images:
        img = PIL.Image.open(dataset_dir+"/"+image_label+"/"+image)
        img = img.resize((64, 64))
        dataset.append((numpy.asarray(img),image_label))

# Multidimensional array
x=[]
y=[]
for  input,image_label in dataset:
    x.append(input)
    y.append(label.index(image_label))
x=numpy.array(x)
y=numpy.array(y)
x_train,y_train=x,y
data_set=(x_train,y_train)
x_train = x_train / 255.0
y_binary = np_utils.to_categorical(y_train)

# Create model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(y_binary.shape[1], activation='softmax'))

# Compile
epochs = 10
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)]

# Fit
model.fit(x_train, y_train, epochs=epochs, batch_size=32,shuffle=True,callbacks=callbacks)

# Evaluation
scores = model.evaluate(x_train, y_train, verbose=0)

# Save model
model.save_weights("model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print(model.summary())
print("Done.")
