from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctcdecode import CTCBeamDecoder
from pydub import AudioSegment
import soundfile as sf
import yaml
import torch
import numpy as np
import json
import os

# SAVED_MODEL_PATH = "wav2vec2-conformer-large"
MAX_LENGTH = 160000
SAMPLE_RATE = 16000


class Wav2vec2PredictServices:
    model = None
    processor = None
    vocab = None
    ctc_lm_params = None
    kenlm_ctcdecoder = None
    _instance = None

    def predict(self, file_path):
        """
        :param file_path: Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the models
        """

        # extract MFCC
        signal = self.preprocess(file_path)
        print(np.shape(signal))
        if len(signal) >= MAX_LENGTH:
            signal = signal[:MAX_LENGTH]
        inputs = self.processor(signal, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        # get the predicted label
        beam_results, beam_scores, timesteps, out_lens = self.kenlm_ctcdecoder.decode(logits)
        pred_with_lm = "".join(self.vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
        predicts = pred_with_lm.strip()

        print("pred_label: ", predicts)
        # predictions = self.models.predict([signal],decoder = 'beam', beam_size = 5)
        # print(predictions)
        # predicted_keyword = predictions[0]
        return predicts

    def preprocess(self, file_path):
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
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
            audio.export(file_name, format="wav")
            signal, sample_rate = sf.read(file_name)
            return signal
        else:
            signal, sample_rate = sf.read(file_path)
            return signal


def init_services():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if Wav2vec2PredictServices._instance is None:
        Wav2vec2PredictServices._instance = Wav2vec2PredictServices()
        Wav2vec2PredictServices.model = Wav2Vec2ForCTC.from_pretrained(
            "./models/w2v_xlsr_model/checkpoint-54075")
        Wav2vec2PredictServices.processor = Wav2Vec2Processor.from_pretrained(
            "./models/w2v_xlsr_model/checkpoint-54075")

        Wav2vec2PredictServices.vocab = Wav2vec2PredictServices.processor.tokenizer.convert_ids_to_tokens(
            range(0, Wav2vec2PredictServices.processor.tokenizer.vocab_size))
        space_ix = Wav2vec2PredictServices.vocab.index('|')
        Wav2vec2PredictServices.vocab[space_ix] = ' '
        with open(os.path.join("./lm", "config_ctc.yaml"),
                  'r') as config_file:
            Wav2vec2PredictServices.ctc_lm_params = yaml.load(config_file, Loader=yaml.FullLoader)
        Wav2vec2PredictServices.kenlm_ctcdecoder = CTCBeamDecoder(Wav2vec2PredictServices.vocab,
                                                                  model_path="./lm/malay_lm.bin",
                                                                  alpha=Wav2vec2PredictServices.ctc_lm_params[
                                                                         'alpha'],
                                                                  beta=Wav2vec2PredictServices.ctc_lm_params[
                                                                         'beta'],
                                                                  cutoff_top_n=40,
                                                                  cutoff_prob=1.0,
                                                                  beam_width=100,
                                                                  num_processes=4,
                                                                  blank_id=Wav2vec2PredictServices.processor.tokenizer.pad_token_id,
                                                                  log_probs_input=True
                                                                  )
    return Wav2vec2PredictServices._instance


if __name__ == "__main__":
    # create 2 instances of the keyword spotting service
    wps = init_services()
    wps1 = init_services()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert wps is wps1

    # make a prediction
    word = wps.predict("test1.wav")
    print(word)
