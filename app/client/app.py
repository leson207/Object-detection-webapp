import time
import base64
import numpy as np
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import prometheus_client

from gRPC import _grcp_client_request
from RESTful import _rest_client_request

import joblib
import tensorflow as tf

app = FastAPI()

app.mount("/metrics", prometheus_client.make_asgi_app())
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=['*'],
    allow_headers=['*'],
)

transform=joblib.load('app/client/test_transform.pkl')
app.mount("/static", StaticFiles(directory="app/client/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=open("app/client/template/index.html").read())

@app.post("/predict")
async def predict_route(request: dict = Body(...)):
    b64_image = request['image']

    process_input(b64_image)
    time.sleep(1)
    return [b64_image]

def process_pred(pred):
    pass
def process_input(input):
    image_data=base64.b64decode(input)
    pil_image = Image.open(BytesIO(image_data)).convert('RGB')
    image = np.array(pil_image)

    image=transform(image=image, bboxes=[], labels=[])['image']

    image=tf.constant(image, dtype=tf.float32)
    pred=_grcp_client_request(image[None])
    
    # pred=_rest_client_request(image[None])
    
    process_pred(pred)