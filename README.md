# tf-serving
Request generation for tf serving using both http and grpc 


Dataset used can be downloaded from the link below

https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

To start tf-server with docker

<b>docker run -p 8501:8501 -p 8500 --mount type=bind,source=< Path to the SavedModel directory>,target=<Destination where we want our model to be on docker container/{ModelName}> -e MODEL_NAME={ModelName} -t tensorflow/serving &</b>


