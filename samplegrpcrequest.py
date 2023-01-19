# %%
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorboard.compat.proto import types_pb2


# %%
CLASS_LABELS = {'Bean': 0, 'Bitter_Gourd': 1, 'Bottle_Gourd': 2, 'Brinjal': 3, 'Broccoli': 4,
 'Cabbage': 5, 'Capsicum': 6, 'Carrot': 7, 'Cauliflower': 8, 'Cucumber': 9, 'Papaya': 10,
 'Potato': 11, 'Pumpkin': 12, 'Radish': 13, 'Tomato': 14}
## inverse keys and labels 

idx_to_labels = {v: k for k, v in CLASS_LABELS.items()}

# %%

img = tf.io.read_file('/home/satish/samples/vegetableDataset/Vegetable Images/test/Capsicum/1041.jpg')
img = tf.io.decode_jpeg(img,channels=3)
img = tf.cast(img,tf.float32)/255
img = tf.expand_dims(img, axis=0)

# %%
channel = grpc.insecure_channel('0.0.0.0:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'vegetable' # this is the name of the model in docker container

# check the inputs of the model using the url 
#http://localhost:8501/v1/models/vegetable/metadata
request.inputs['conv2d_input'].CopyFrom(tf.make_tensor_proto(img,shape=img.shape))

resp = stub.Predict(request, 10.0)

#print(resp)

#%%
class_index = np.argmax(resp.outputs['dense_1'].float_val)
#convert  class_index from np.int64 to int
class_index = int(class_index)
conf = resp.outputs['dense_1'].float_val[class_index]
class_predicted = idx_to_labels[class_index]
print(class_index)
print(class_predicted)
print(conf)

# %%
