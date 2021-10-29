import random
import os
from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
from pydantic import BaseModel
from predicts.wav2vec2_predict_services import init_services
from fastapi.encoders import jsonable_encoder

app = FastAPI(title='Malay-Speech Recognition', version='1.0',
              description='wav2vec2 models is used for prediction')  #


# create class to receive request from client
class Data(BaseModel):
    model: str
    lm: str


# create class to multi choices about model
class ModelPattern:
    def __init__(self, model, lm):
        self.model = model
        self.lm = lm

    def __setitem__(self, model, lm):
        self.model = model
        self.lm = lm

    def __getitem__(self):
        return {"model": self.model, "lm": self.lm}


AUDIO_DIR = "./api/audio"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# init instance ModelPattern
model_pattern = ModelPattern("model1", "CTC + 4-gram")


@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}


@app.post("/predict")
def up_file(file: UploadFile = File(...)):
    """Endpoint to predict keyword
        :return (json):  json file with the following format:
            {
                "file_name": "24567.wav"
            }
    """
    # get file from POST request and save it
    if ".wav" in file.filename:
        file_name = str(random.randint(0, 100000)) + ".wav"
    elif ".mp3" in file.filename:
        file_name = str(random.randint(0, 100000)) + ".mp3"
    else:
        return jsonable_encoder("Audio format error")
    with open(os.path.join(AUDIO_DIR, file_name), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # take path_file to predict
    cache_file = os.path.join(AUDIO_DIR, file_name)

    # take model vs lm to predict
    pattern_dict = model_pattern.__getitem__()

    # operate prediction
    wps = init_services(pattern_dict["model"])
    predicted_word = wps.predict(cache_file, pattern_dict)

    # return result for client
    result = {"word": predicted_word}
    return jsonable_encoder(result)


@app.post("/pattern")
def predict_pattern(file: Data):
    """
    :param file: (dict) contain name model & name lm are used for predict
    :return: result (dict) for make sure request ok
    """
    data = file.dict()
    model_pattern.__setitem__(data["model"], data["lm"])
    result = {"result": model_pattern.__getitem__()}
    return jsonable_encoder(result)


if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
