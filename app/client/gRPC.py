import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def _grcp_client_request(inputs,url='server:8500',model_name='yolo',timeout=10):

    channel =grpc.insecure_channel(url)
    stub=prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name='serving_default'

    request.inputs["args_0"].CopyFrom(
        tf.make_tensor_proto(
            inputs,
            dtype=np.float32,
            shape=inputs.shape
        )
    )

    response = stub.Predict(request, timeout=timeout)
    output_0= np.array(response.outputs['output_0'].float_val).reshape(output_shape[0])
    output_1= np.array(response.outputs['output_1'].float_val).reshape(output_shape[1])
    output_2= np.array(response.outputs['output_2'].float_val).reshape(output_shape[2])
    return output_0,output_1,output_2

num_classes=208
output_shape= [
    [-1, 3, 7, 7, num_classes+5],
    [-1, 3, 14, 14, num_classes+5],
    [-1, 3, 28, 28, num_classes+5]
]