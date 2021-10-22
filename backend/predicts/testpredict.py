import os

os.environ['MALAYA_CACHE'] = '/Users/huseinzolkepli/Documents/malaya-speech-cache'

import malaya_speech
import numpy as np
import soundfile as sf
import IPython.display as ipd
#from playsound import playsound

def predict(path):

    model_wav2vec = malaya_speech.stt.deep_ctc(model='wav2vec2-conformer-large')
    signal, sr = sf.read(path)
    return model_wav2vec.predict([signal], decoder = 'beam', beam_size = 5)

if __name__ == "__main__":
    word = predict('test1.wav')
    ipd.display('test1.wav')
    print(word[0])