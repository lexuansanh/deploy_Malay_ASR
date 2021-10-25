import random
import os
from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn
import shutil

from pydantic import BaseModel

from predicts.wav2vec2_predict_services import init_services
from fastapi.encoders import jsonable_encoder

app = FastAPI(title='Malay-Speech Recognition', version='1.0',
              description='wav2vec2 models is used for prediction')  #


class Data(BaseModel):
    model: str
    lm: str


class Arg:
    def __init__(self, model, lm):
        self.model = model
        self.lm = lm

    def __setitem__(self, model, lm):
        self.model = model
        self.lm = lm

    def __getitem__(self):
        return {"model": self.model, "lm": self.lm}


# init_prediction_services:


AUDIO_DIR = "./audio"
arg = Arg("model1", "4-gram")

@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}


@app.post("/predict_old")
def predict(file: Data):
    """Endpoint to predict keyword
        :return (json): This endpoint returns a json file with the following format:
            {
                "word": "satu dua tiga"
            }
    """
    data = file.dict()
    # get file from POST request and save itpp
    # file_name = str(random.randint(0, 100000))
    # with open(file_name, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    # instantiate keyword spotting service singleton and get prediction
    file_name = os.path.join(AUDIO_DIR, data["file_name"])
    wps = init_services()
    predicted_word = wps.predict(file_name)

    # we don't need the audio file any more - let's delete it!
    # os.remove(file_name)

    # send back result as a json file
    result = {"word": predicted_word}

    return jsonable_encoder(result)


@app.post("/predict")
def up_file(file: UploadFile = File(...)):
    """Endpoint to predict keyword
        :return (json):  json file with the following format:
            {
                "file_name": "24567.wav"
            }
    """

    # get file from POST request and save itpp
    file_name = ""
    if ".wav" in file.filename:
        file_name = str(random.randint(0, 100000)) + ".wav"
    elif ".mp3" in file.filename:
        file_name = str(random.randint(0, 100000)) + ".mp3"
    else:
        return jsonable_encoder("Audio format error")
    with open(os.path.join(AUDIO_DIR, file_name), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cache_file = os.path.join(AUDIO_DIR, file_name)
    wps = init_services()
    arg_dict = arg.__getitem__()
    predicted_word = wps.predict(cache_file, arg_dict)

    # instantiate keyword spotting service singleton and get prediction
    result = {"word": predicted_word}

    return jsonable_encoder(result)


@app.post("/arg")
def predict_arg(file: Data):
    data = file.dict()
    arg.__setitem__(data["model"], data["lm"])
    result = {"result": arg.__getitem__()}
    return jsonable_encoder(result)


if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
