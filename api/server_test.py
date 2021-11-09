import random
import os
import json
from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
from pydantic import BaseModel
from pyngrok import ngrok, conf
from predicts.wav2vec2_predict_services import init_services
from fastapi.encoders import jsonable_encoder
from colabcode import ColabCode
from starlette.responses import RedirectResponse
app = FastAPI(title='Malay-Speech Recognition', version='1.0',
              description='wav2vec2 models is used for prediction')  #


#Set account authtoken for ngrok
NGROK_AUTH_TOKEN = "20PNthCnmPxXyuT1KvRskAphbuw_5PnGX4kX6VbLqXZwTRZP9"
URL_FOLDER = "./api/url"
AUDIO_DIR = "./api/audio"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)
if not os.path.exists(URL_FOLDER):
    os.makedirs(URL_FOLDER)


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


# init instance ModelPattern
model_pattern = ModelPattern("model1", "CTC + 4-gram")


@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    active_tunnels = ngrok.get_tunnels()
    public_url = {"url": active_tunnels[0].public_url}
    with open(os.path.join(URL_FOLDER, "url.json"), "w") as bf:
      json.dump(public_url, bf)
    return RedirectResponse(url = "https://malayasr.herokuapp.com")

@app.get('/url')
def get_url():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    active_tunnels = ngrok.get_tunnels()
    public_url = {"url": active_tunnels[0].public_url}
    with open(os.path.join(URL_FOLDER, "url.json"), "w") as bf:
      json.dump(public_url, bf)
    return RedirectResponse(url = "https://malayasr.herokuapp.com")

@app.get('/redirect')
def re_direct():
    """
     Home endpoint which can be used to test the availability of the application.
     """
      
    return RedirectResponse(url = "https://malayasr.herokuapp.com")


@app.post("/predict")
def up_file(file: UploadFile = File(...)):
    """Endpoint to predict keyword
        :return (json):  json file with the following format:
            {
                "file_name": "24567.wav"
            }
    """
    # get file from POST request and save it
    if ".wav" in file.filename.lower():
        file_name = str(random.randint(0, 100000)) + ".wav"
    elif ".mp3" in file.filename.lower():
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
    result = {"word_OK_": predicted_word}
    return jsonable_encoder(result)


@app.post("/pattern")
def predict_pattern(file: Data):
    """
    :param file: (dict) contain name model & name lm are used for predict
    :return: result (dict) for make sure request ok
    """
    data = file.dict()
    model_pattern.__setitem__(data["model"], data["lm"])
    result = {"result_OK_": model_pattern.__getitem__()}
    return jsonable_encoder(result)

#----------------------------------------------------------------

import random
from audio_processing import extract_audio

class AutoSubPattern:
    def __init__(self, method, version):
        self.method = method
        self.version = version

    def __setitem__(self, method, version):
        self.method = method
        self.version = version

    def __getitem__(self):
        return {"method": self.method, "version": self.version}

autosub_pattern = AutoSubPattern("upload_video", "basic_version")

def calc_checksum():
    hash = random.getrandbits(128)
    return f"%016x" % hash
class AutoSubVersion(BaseModel):
    method: str
    version: str

@app.post("/autosub_version")
async def get_autosub_version(received_data: AutoSubVersion):
    try:
        received_data_version = received_data.dict()
        autosub_pattern.__setitem__(received_data_version["method"], received_data_version["version"])
        result = {"result_OK_": autosub_pattern.__getitem__()}

        return jsonable_encoder(result)

    except Exception as e:
        print(f"{e}")
        return None

# class AutoSubGeneration(BaseModel):
#     token: str
#     video_file: UploadFile = File(...)
VIDEO_AUTOSUB_DIR = "./api/autosub_data/video"
AUDIO_AUTOSUB_DIR = "./api/autosub_data/audio"
OUTPUT_AUTOSUB_DIR = "./api/autosub_data/output"

if not os.path.exists(VIDEO_AUTOSUB_DIR):
    os.makedirs(VIDEO_AUTOSUB_DIR)

if not os.path.exists(AUDIO_AUTOSUB_DIR):
    os.makedirs(AUDIO_AUTOSUB_DIR)

if not os.path.exists(OUTPUT_AUTOSUB_DIR):
    os.makedirs(OUTPUT_AUTOSUB_DIR)

import datetime

def get_timestamp_string(timedelta, format='srt'):
    """Convert the timedelta into something that can be used by a subtitle file.

    Args:
        timedelta : timedelta timestmap
        format : subtitle format
    """
    sep = '.' if format == "vtt" else ','
    # timedelta may be eg, '0:00:14'
    if '.' in str(timedelta):
        timestamp = "0" + str(timedelta).split(".")[0] + sep + str(timedelta).split(".")[-1][:3]
    else:
        timestamp = "0" + str(timedelta) + sep + "000"
    return timestamp


def generate_one_alignment(line_count, text, time_start, time_end):
    start_time = datetime.timedelta(seconds=float(time_start))
    end_time = datetime.timedelta(seconds=float(time_end))
    
    _time_start = get_timestamp_string(start_time)
    _time_end = get_timestamp_string(end_time)

    txt = f"{str(line_count)}\n{_time_start} --> {_time_end}\n{text}\n\n"
    return txt


def generate_many_alignment(alignment_dict):
    print("generate_many_alignment: ", alignment_dict)
    text_results = alignment_dict['results']
    alignment_lst = alignment_dict['predict_alignment']

    autosub_result = ""

    line_count = 0
    for text, alignment in zip(text_results, alignment_lst):
        time_start, time_end = alignment
        # print(time_start, time_end)
        txt = generate_one_alignment(line_count, text, time_start, time_end)
        autosub_result += txt
        line_count += 1

    return autosub_result

def generate_srt_file(autosub_result, srt_file_path):
    try:
        with open(srt_file_path, "w") as file_handle:
            file_handle.write(autosub_result)
            print("Generate srt file to {}".format(srt_file_path))

    except Exception as e:
        print(f"{e}")

import subprocess

def attach_subtitle_to_video(video_file, srt_file, output_video_file):
    cmd = f"ffmpeg -i {video_file} -i {srt_file} -c copy -c:s mov_text {output_video_file}"
    ret = subprocess.call(cmd, shell=True)
    print("Attach subtitle to video saved to {}".format(output_video_file))

from fastapi.responses import FileResponse

@app.get("/autosub_transcript_generation")
async def post_autosub_transcript_generation(file: UploadFile = File(...)):

    checksum_generation = calc_checksum()

    file_name = None
    if ".mp4" in file.filename.lower():
        file_name = f"{checksum_generation}.mp4"
    else:
        return jsonable_encoder("Video format error")
    with open(os.path.join(VIDEO_AUTOSUB_DIR, file_name), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = None
    # take path_file to predict
    if file_name is not None:
        video_cache_file = os.path.join(VIDEO_AUTOSUB_DIR, file_name)
        audio_cache_file = os.path.join(AUDIO_AUTOSUB_DIR, file_name.replace('.mp4', '.wav'))

        try:
            # extract audio from video file
            extract_audio(video_cache_file, audio_cache_file)

            # take model vs lm to predict
            pattern_dict = model_pattern.__getitem__()
            autosub_dict = autosub_pattern.__getitem__()

            wps = init_services(pattern_dict["model"])
            print("autosub dict: ", autosub_dict)
            if autosub_dict['version'] == 'advance_version':
                new_video_file_name = calc_checksum()
                new_video_file_name = f"{new_video_file_name}.mp4"
                # print(new_video_file_name)
                output_video_file = os.path.join(VIDEO_AUTOSUB_DIR, new_video_file_name)
                # print(output_video_file)
                srt_file_path = os.path.join(OUTPUT_AUTOSUB_DIR, file_name.replace('.mp4', '.srt'))
                # print("audio_file: ", audio_cache_file)
                predicted_word, autosub_prediction = wps.predict(audio_cache_file, pattern_dict, True, True)
                
                # print("autosub_prediction: ", autosub_prediction)
                autosub_result = generate_many_alignment(autosub_prediction)

                # print("Generating ", autosub_result)
                generate_srt_file(autosub_result, srt_file_path)

                # print("Generating ", output_video_file)

                attach_subtitle_to_video(video_cache_file, srt_file_path, output_video_file)

                if os.path.exists(output_video_file):
                    result = {"word_OK_": predicted_word, 'path': output_video_file}
                    # return FileResponse(output_video_file, media_type='text/mp4')

            else:
                # operate prediction
                predicted_word = wps.predict(audio_cache_file, pattern_dict)

                # return result for client
                result = {"word_OK_": predicted_word, "path": ""}

            print(result)
            if result is not None:
                return jsonable_encoder(result)
            else:
                return None

        except Exception as e:
            print(f"{e}")
            return None

class AutoSubVersion(BaseModel):
    version: str
    method: str

class AutoSubYouTube(BaseModel):
    url: str

from glob import glob

def download_youtube_video(url, only_audio=False):
    from pytube import YouTube 

    file_name = calc_checksum()
    yt = YouTube(url.strip())
    video = yt.streams

    _video = None
    output_filename_path = ""

    if only_audio:
        output_filename_path = os.path.join(AUDIO_AUTOSUB_DIR, file_name)
        _video = video.filter(only_audio=True).first()

    else:
        output_filename_path = os.path.join(VIDEO_AUTOSUB_DIR, file_name)
        _video = video.get_lowest_resolution()
    
    if _video is not None and output_filename_path != "":
        _video.download(output_path=output_filename_path)
        if os.path.isdir(output_filename_path):
            src_file_name = glob(f"{output_filename_path}/*.mp4")[0]
            new_file_name = f"{output_filename_path}/{file_name}.mp4"
            os.rename(src_file_name, new_file_name)

        return new_file_name
    else:
        return None

@app.get("/autosub_youtube")
async def get_autosub_youtube(file: AutoSubYouTube):

    received_url = file.dict()['url']
    print(received_url)
    file_name = None
    if received_url != "":
        file_name = download_youtube_video(received_url)
    else:
        return jsonable_encoder("Video format error")

    result = None
    # take path_file to predicts
    if file_name is not None:
        video_cache_file = file_name
        audio_cache_file = file_name.replace('.mp4', '.wav')

        try:
            # extract audio from video file
            extract_audio(video_cache_file, audio_cache_file)

            # take model vs lm to predict
            pattern_dict = model_pattern.__getitem__()
            autosub_dict = autosub_pattern.__getitem__()

            wps = init_services(pattern_dict["model"])

            if autosub_dict['version'] == 'advance_version':
                new_video_file_name = calc_checksum()
                new_video_file_name = f"{new_video_file_name}.mp4"

                output_video_file = os.path.join(VIDEO_AUTOSUB_DIR, new_video_file_name)

                srt_file_path = os.path.join(OUTPUT_AUTOSUB_DIR, new_video_file_name.replace('.mp4', '.srt'))
                
                predicted_word, autosub_prediction = wps.predict(audio_cache_file, pattern_dict, True, True)
                
                # print("autosub_prediction: ", autosub_prediction)
                autosub_result = generate_many_alignment(autosub_prediction)

                # print("Generating ", autosub_result)
                generate_srt_file(autosub_result, srt_file_path)

                # print("Generating ", output_video_file)

                attach_subtitle_to_video(video_cache_file, srt_file_path, output_video_file)

                if os.path.exists(output_video_file):
                    result = {"word_OK_": predicted_word, 'path': output_video_file}

            else:
                # operate prediction
                predicted_word = wps.predict(audio_cache_file, pattern_dict)

                # return result for client
                result = {"word_OK_": predicted_word, "path": ""}

            print(result)

            if result is not None:
                return jsonable_encoder(result)
            else:
                return None

        except Exception as e:
            print(f"{e}")
            return None

from starlette.responses import FileResponse

@app.get('/download_video_file')
async def download_video_file(file: AutoSubYouTube):
    print("file")
    file_location = file['path']
    if os.path.exists(file_location):
        return FileResponse(file_location)
    else:
        return None


if __name__ == '__main__':
    server = ColabCode(port=10000, password="sanhlx@fsoft", authtoken=NGROK_AUTH_TOKEN, mount_drive=True, code = False)
    
    server.run_app(app=app)
    #uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)