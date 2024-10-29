import json
import requests
import numpy as np

def _rest_client_request(inputs,url='server:8501'):
    data=json.dumps({
        'signature_name': 'serving_default',
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

num_classes=208
output_shape= [
    [-1, 3, 7, 7, num_classes+5],
    [-1, 3, 14, 14, num_classes+5],
    [-1, 3, 28, 28, num_classes+5]
]