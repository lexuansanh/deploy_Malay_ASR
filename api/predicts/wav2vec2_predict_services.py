from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctcdecode import CTCBeamDecoder
from pydub import AudioSegment
import soundfile as sf
import yaml
import torch
import numpy as np
import json
import os
from time import time
import librosa

# SAVED_MODEL_PATH = "wav2vec2-conformer-large"
MAX_LENGTH = 160000
SAMPLE_RATE = 16000


# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Wav2vec2PredictServices:
    model = {}
    processor = {}
    vocab = {}
    ctc_lm_params = {}
    kenlm_ctcdecoder = {}
    _instance = {}

    def predict(self, file_path, pattern_dict):
        """
        :param file_path: Path to audio file to predict
        :param pattern_dict: some options about models and lms
        :return predicted_keyword (str): Keyword predicted by the models
        """

        start_time = time()
        # Choice model for prediction
        processor = self.processor[pattern_dict["model"]]
        model = self.model[pattern_dict["model"]]
        vocab = self.vocab[pattern_dict["model"]]
        kenlm_ctcdecoder = self.kenlm_ctcdecoder[pattern_dict["model"]]

        # preprocess audio file
        audio_data = self._preprocess(file_path)
        array_signal = self._split_array(audio_data)
        results = ""

        # predict with model and decode
        for signal in array_signal:
            predicts = ""
            inputs = processor(signal, sampling_rate=16_000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

            # get the predicted label
            if pattern_dict["lm"] == 'CTC':
                predicted_ids = torch.argmax(logits, dim=-1)
                predicts = processor.decode(predicted_ids[0])
            elif pattern_dict["lm"] == 'CTC + 4-gram':
                beam_results, beam_scores, timesteps, out_lens = kenlm_ctcdecoder.decode(logits)
                pred_with_lm = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
                predicts = pred_with_lm.strip()
            results += predicts
            results += " "
            print("pred_label: ", predicts)
        predict_time = time() - start_time
        # results += f" (predict_time: {predict_time} s)"

        return results.strip()

    def _preprocess(self, file_path):
        """Extract MFCCs from audio file.
        :param file_path: (str): Path of audio file
        :return signal: (np.array): array of speech signal
        """
        # load audio file
        if ".mp3" in file_path:
            print(file_path)
            audio = AudioSegment.from_file(file_path)
            file_name = file_path.replace(".mp3", ".wav")
            print(file_name)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(file_name, format="wav")
            signal, sample_rate = librosa.load(file_name, sr=SAMPLE_RATE, mono=True)
            print("sample_rate", sample_rate)
            #signal = self._resample_if_necessary(signal, sample_rate)
            return signal
        else:
            signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            # signal = self._resample_if_necessary(signal, sample_rate)
            return signal

    @staticmethod
    def _resample_if_necessary(signal, sr):
        """
        :param signal: audio data (np.array or list)
        :param sr: sampling rate (int)
        :return: signal(np.array) with fixed sampling rate(16000)
        """

        if sr != SAMPLE_RATE:
            signal = librosa.resample(signal, sr, SAMPLE_RATE)
        return signal

    @staticmethod
    def _split_array(audio_data):
        """Extract MFCCs from audio file.
        :param : audio_data: Data(np.array) of audio file
        :return array_signal: list[(np.array)]: list arrays of audio_data
        """
        # load audio file
        array_signal = []
        n_split = len(audio_data) // MAX_LENGTH
        for i in range(n_split):
            signal = audio_data[i * MAX_LENGTH:(i + 1) * MAX_LENGTH]
            array_signal.append(signal)
        if len(audio_data) % MAX_LENGTH != 0:
            array_signal.append(audio_data[n_split * MAX_LENGTH:])
            return array_signal


def init_services(model_pattern):
    """Factory function for Wav2vec2PredictServices class.
    :return Wav2vec2PredictServices._instance (model_pattern):
    """
    # ensure an instance is created only the first time the factory function is called
    if model_pattern not in Wav2vec2PredictServices._instance.keys():
        Wav2vec2PredictServices._instance[model_pattern] = Wav2vec2PredictServices()
        Wav2vec2PredictServices.model[model_pattern] = Wav2Vec2ForCTC.from_pretrained(
            f"./api/models/{model_pattern}")
        Wav2vec2PredictServices.processor[model_pattern] = Wav2Vec2Processor.from_pretrained(
            f"./api/models/{model_pattern}")

        Wav2vec2PredictServices.vocab[model_pattern] = Wav2vec2PredictServices.processor[
            model_pattern].tokenizer.convert_ids_to_tokens(
            range(0, Wav2vec2PredictServices.processor[model_pattern].tokenizer.vocab_size))
        space_ix = Wav2vec2PredictServices.vocab[model_pattern].index('|')
        Wav2vec2PredictServices.vocab[model_pattern][space_ix] = ' '
        with open(os.path.join(f"./api/models/{model_pattern}/lm", "config_ctc.yaml"),
                  'r') as config_file:
            Wav2vec2PredictServices.ctc_lm_params[model_pattern] = yaml.load(config_file, Loader=yaml.FullLoader)
        Wav2vec2PredictServices.kenlm_ctcdecoder[model_pattern] = CTCBeamDecoder(
            Wav2vec2PredictServices.vocab[model_pattern],
            model_path=f"./api/models/{model_pattern}/lm/malay_lm.bin",
            alpha=Wav2vec2PredictServices.ctc_lm_params[model_pattern][
                'alpha'],
            beta=Wav2vec2PredictServices.ctc_lm_params[model_pattern][
                'beta'],
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=100,
            num_processes=4,
            blank_id=Wav2vec2PredictServices.processor[model_pattern].tokenizer.pad_token_id,
            log_probs_input=True
        )
    return Wav2vec2PredictServices._instance[model_pattern]


if __name__ == "__main__":
    # create 2 instances of the Wav2vec2PredictServices
    wps = init_services("model1")
    wps1 = init_services("model1")

    # check that different instances of the Wav2vec2PredictServices point back to the same object (singleton)
    assert wps is wps1

    # make a prediction
    word = wps.predict("test1.wav", {"model": "model1", "lm": "4-gram"})
    print(word)
