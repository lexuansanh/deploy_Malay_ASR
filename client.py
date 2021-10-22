import requests
import time
import os
# server url
URL = "http://0.0.0.0:8080/predict"


# audio file we'd like to send for predicting keyword/home/sanhlx/PycharmProjects/wav2vec2-malay/tests
#FILE_PATH = "/home/sanhlx/PycharmProjects/wav2vec2-malay/tests/waves/fp6ABAjcXPI.0008.wav"
file = input("enter file name: ")
FILE_PATH = os.path.join("./tests/waves", file)
if __name__ == "__main__":
    start = time.time()
    # open files
    file = open(FILE_PATH, "rb")

    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()
    stop = time.time()
    print("Predicted word: {}".format(data["word"]))
    print("pred Time: %0.2f s" %(stop-start))