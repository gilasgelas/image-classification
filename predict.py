import os
import numpy
import keras
import PIL
import pickle

classifier_f = open("labels.txt", "rb")
labelMapper = pickle.load(classifier_f)
classifier_f.close()

# Prepare model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

# Prepare image
test_dir="test"
images=os.listdir(test_dir)
img=PIL.Image.open(test_dir+"/"+images[0])
img=img.resize((64,64))
image=numpy.array([numpy.asarray(img)])
image = image.astype('float32')
image = image / 255.0

# Predict
result=loaded_model.predict(image)
for i in range(len(labelMapper)):
    print(labelMapper[i]+": "+str(result[0][i]))
