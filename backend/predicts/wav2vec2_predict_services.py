#import malaya_speech
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctcdecode import CTCBeamDecoder
import soundfile as sf
import yaml
import torch
import numpy as np
import json
import os

# SAVED_MODEL_PATH = "wav2vec2-conformer-large"
MAX_LENGTH = 22050
SAMPLE_RATE = 16000

class _Wav2vec2_Predict_Services:

    model = None
    processor = None
    vocab = None
    ctc_lm_params = None
    kenlm_ctcdecoder = None
    _instance = None

    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        signal = self.preprocess(file_path)
        inputs = self.processor(signal, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        # get the predicted label
        beam_results, beam_scores, timesteps, out_lens = self.kenlm_ctcdecoder.decode(logits)
        pred_with_lm = "".join(self.vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
        predicts = pred_with_lm.strip()

        print("pred_label: ", predicts)
        # predictions = self.model.predict([signal],decoder = 'beam', beam_size = 5)
        # print(predictions)
        # predicted_keyword = predictions[0]
        return predicts


    def preprocess(self, file_path):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        """

        # load audio file
        signal, sample_rate = sf.read(file_path)

        return signal


def Wav2vec2_Predict_Services():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Wav2vec2_Predict_Services._instance is None:
        _Wav2vec2_Predict_Services._instance = _Wav2vec2_Predict_Services()
        _Wav2vec2_Predict_Services.model = Wav2Vec2ForCTC.from_pretrained("/home/sanhlx/PycharmProjects/wav2vec2-malay/deploy_web/models/w2v_xlsr_model/checkpoint-54075")
        _Wav2vec2_Predict_Services.processor = Wav2Vec2Processor.from_pretrained("/home/sanhlx/PycharmProjects/wav2vec2-malay/deploy_web/models/w2v_xlsr_model/checkpoint-54075")

        _Wav2vec2_Predict_Services.vocab = _Wav2vec2_Predict_Services.processor.tokenizer.convert_ids_to_tokens(range(0, _Wav2vec2_Predict_Services.processor.tokenizer.vocab_size))
        space_ix = _Wav2vec2_Predict_Services.vocab.index('|')
        _Wav2vec2_Predict_Services.vocab[space_ix] = ' '
        with open(os.path.join("/home/sanhlx/PycharmProjects/wav2vec2-malay/deploy_web/lm", "config_ctc.yaml"), 'r') as config_file:
            _Wav2vec2_Predict_Services.ctc_lm_params = yaml.load(config_file, Loader=yaml.FullLoader)
        _Wav2vec2_Predict_Services.kenlm_ctcdecoder = CTCBeamDecoder(_Wav2vec2_Predict_Services.vocab,
                                          model_path="/home/sanhlx/PycharmProjects/wav2vec2-malay/deploy_web/lm/malay_lm.bin",
                                          alpha=_Wav2vec2_Predict_Services.ctc_lm_params['alpha'],
                                          beta=_Wav2vec2_Predict_Services.ctc_lm_params['beta'],
                                          cutoff_top_n=40,
                                          cutoff_prob=1.0,
                                          beam_width=100,
                                          num_processes=4,
                                          blank_id=_Wav2vec2_Predict_Services.processor.tokenizer.pad_token_id,
                                          log_probs_input=True
                                          )
    return _Wav2vec2_Predict_Services._instance


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    wps = Wav2vec2_Predict_Services()
    wps1 = Wav2vec2_Predict_Services()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert wps is wps1

    # make a prediction
    word = wps.predict("test1.wav")
    print(word)