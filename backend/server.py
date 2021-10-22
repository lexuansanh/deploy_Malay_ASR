import random
import os
from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn
import shutil
from predicts.wav2vec2_predict_services import init_services
from fastapi.encoders import jsonable_encoder

app = FastAPI(title='Malay-Speech Recognization', version='1.0',
              description='wav2vec2 models is used for prediction')  #


# class Data(BaseModel):
#     file_path: FilePath

@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Endpoint to predict keyword
        :return (json): This endpoint returns a json file with the following format:
            {
                "word": "satu dua tiga"
            }
    	"""

    # get file from POST request and save itpp
    file_name = str(random.randint(0, 100000))
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # instantiate keyword spotting service singleton and get prediction
    wps = init_services()
    predicted_word = wps.predict(file_name)

    # we don't need the audio file any more - let's delete it!
    os.remove(file_name)

    # send back result as a json file
    result = {"word": predicted_word}

    return jsonable_encoder(result)


@app.post("/multipredict")
def predict(files: List[UploadFile] = File(...)):
    """Endpoint to predict keyword
        :return (json): This endpoint returns a json file with the following format:
            [
            {
                "word": "satu dua tiga"
            },
            {
                "word": "lima enam empat"
            }
            ]
    	"""

    wps = init_services()
    # get file from POST request and save itpp
    results = []
    for data in files:
        file_name = str(random.randint(0, 100000))
        with open(file_name, "wb") as buffer:
            shutil.copyfileobj(data.file, buffer)

        # instantiate keyword spotting service singleton and get prediction

        predicted_word = wps.predict(file_name)

        # we don't need the audio file any more - let's delete it!
        os.remove(file_name)

        # send back result as a json file
        result = {"word": predicted_word}
        results.append(result)

    return jsonable_encoder(results)


if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
