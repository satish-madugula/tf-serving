# %%
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPool2D,Dense,activation,Activation,Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as pyplot

train_datagen = ImageDataGenerator(rescale=1/255.)
valid_datagen = ImageDataGenerator(rescale=1/255.)

train_gen = train_datagen.flow_from_directory(
    '/home/satish/samples/vegetableDataset/Vegetable Images/train',
    target_size= (224,224),
    batch_size=32,
    class_mode ='categorical'
)


# %%
valid_gen = valid_datagen.flow_from_directory(
    '/home/satish/samples/vegetableDataset/Vegetable Images/validation',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical')


model = keras.models.Sequential()
model.add(Conv2D(32, (3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128, (3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(15,activation='softmax'))

print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(train_gen,epochs=10,steps_per_epoch=100,validation_data=valid_gen,validation_steps=15)


#saving the model and weights to hdf5 file
model.save('vegetable_model.hdf5')

#saving the model in pb format for tfserving using tf.saved_model.save function...
# need to create a folder to save all the details and the required format for the tfserving... 
os.makedirs('vegetable_model', exist_ok=True)

tf.saved_model.save(model, 'vegetable_model/1/')

# %%
