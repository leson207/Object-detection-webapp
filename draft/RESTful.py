import cv2
import json
import joblib
import requests
import numpy as np

def _rest_client_request(url,signature_name,inputs):
    data=json.dumps({
        'signature_name':signature_name,
        'instances': inputs.tolist()
    })
    headers={"Content-Type" : "application/json"}
    response=requests.post(url,data=data,headers=headers)

    response.raise_for_status()

    outputs=json.loads(response.text)['predictions'][0]

    output_0= np.array(outputs['output_0']).reshape(output_shape[0])
    output_1= np.array(outputs['output_1']).reshape(output_shape[1])
    output_2= np.array(outputs['output_2']).reshape(output_shape[2])

    return output_0,output_1,output_2

def prepare_input(file_name):
    image = cv2.imread(file_name)
    transform=joblib.load('artifacts/preprocessor/test_transform.pkl')
    image=transform(image=image, bboxes=[], labels=[])['image']
    return image


num_classes=208
output_shape= [
    [-1, 3, 7, 7, num_classes+5],
    [-1, 3, 14, 14, num_classes+5],
    [-1, 3, 28, 28, num_classes+5]
]

# image=prepare_input('test.jpg')
# result=_rest_client_request('http://localhost:8501/v1/models/yolo:predict',
#                                  'serving_default',image[None])

# print(result[0].shape)
# print(result[1].shape)
# print(result[2].shape)