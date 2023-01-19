# %%
import tensorflow as tf
import json
import requests
import numpy as np

SERVER_URL = 'http://localhost:8501/v1/models/vegetable:predict'
CLASS_LABELS = {'Bean': 0, 'Bitter_Gourd': 1, 'Bottle_Gourd': 2, 'Brinjal': 3, 'Broccoli': 4,
 'Cabbage': 5, 'Capsicum': 6, 'Carrot': 7, 'Cauliflower': 8, 'Cucumber': 9, 'Papaya': 10,
 'Potato': 11, 'Pumpkin': 12, 'Radish': 13, 'Tomato': 14}
## inverse keys and labels 

idx_to_labels = {v: k for k, v in CLASS_LABELS.items()}

# %%
def make_prediction(img):
   data = json.dumps({"signature_name": "serving_default", "instances": img.numpy().tolist()})
   headers = {"content-type": "application/json"}
   json_response = requests.post(SERVER_URL, data=data, headers=headers)
   predictions = json.loads(json_response.text)['predictions']
   return predictions

# %%
img = tf.io.read_file('/home/satish/samples/vegetableDataset/Vegetable Images/test/Capsicum/1041.jpg')
img = tf.io.decode_jpeg(img,channels=3)
img = tf.cast(img,tf.float32)/255
img = tf.expand_dims(img, axis=0)

pred = make_prediction(img)
class_index = np.argmax(pred)
conf = pred[0][class_index]
class_predicted = idx_to_labels[class_index]
print(class_predicted)
print(pred)
print(class_index)
print(conf)
# %%
