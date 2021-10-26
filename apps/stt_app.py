import json
import streamlit as st
from hydralit import HydraHeadApp
import time
import requests
import logging

URL = "http://0.0.0.0:8080"
current_time = time.strftime("%H%M%S-%d%M%y")
file_name = "transcript_" + str(current_time)


class SpeechToTextApp(HydraHeadApp):

    def __init__(self, title="", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        self.logger = logging.getLogger(__name__)

    def run(self):

        try:
            # ----------------------------------------------------------------
            # Show display of Speech To Text app
            st.title("Speech To Text")
            st.subheader("App to transcribe available audio voice to text")
            st.markdown('<br><br>', unsafe_allow_html=True)

            _, col2, _ = st.columns((1, 8, 1))
            self.display_app_header(self.title, True)
            # ----------------------------------------------------------------

            # ----------------------------------------------------------------
            # selection of acoustic model and language model
            arg_return = self.generate_sidebar()

            # upload file audio
            upload_complete, audio_file = self.upload_file(col2)

            # transcribe audio
            trans_btn = col2.button("Transcribe")

            if upload_complete:
                predict_str = None
                if trans_btn:
                    predict_str = self.predict(audio_file)

                if predict_str is not None:
                    transcript_result = col2.text_area("Text Transcript", value=predict_str["word"], height=300)

                    if transcript_result is not None:
                        self.save_transcript(col2, transcript_result)

        except Exception as e:
            st.image("./resources/failure.png", width=100, )
            st.error(
                'An error has occurred, someone will be punished for your inconvenience, we humbly request you try again.')
            st.error('Error details: {}'.format(e))

    def save_transcript(self, col, result):
        if result is not None:
            _result = result.strip()

        try:
            col.download_button(label="Save transcript",
                                data=_result,
                                file_name=f'{file_name}.txt',
                                mime='text/csv')

            col.success("Save transcript successfully")
            self.logger.info("Saved transcript successfully")

        except Exception as e:
            col.error("Error saving transcript: {}".format(e))
            self.logger.error("Error saving transcript: {}".format(e))

    def predict(self, audio_file):
        values = {"file": (audio_file.name, audio_file, "audio/wav")}

        st.session_state.text_result = None

        if isinstance(values, dict):
            try:
                response = requests.post(f"{URL}/predict", files=values)
                st.session_state.text_result = response.json()

                if st.session_state.text_result is not None:
                    self.logger.info(f"Predict: {st.session_state.text_result}")
                else:
                    self.logger.warning("Predict failed")

            except Exception as e:
                st.error('Error details: {}'.format(e))
                self.logger.error(f"Error: {e}")

        else:
            st.warning('Predict failed')
            self.logger.info('Predict failed')

        return st.session_state.text_result

    def generate_sidebar(self):
        if "predict_arg" not in st.session_state:
            st.session_state.predict_arg = {"model": "model1",
                                            "lm": "CTC + 4-gram"}
        if "acoustic_model" not in st.session_state:
            st.session_state.acoustic_model = "model1"
        if "lm" not in st.session_state:
            st.session_state.lm_option = "CTC + 4-gram"

        with st.sidebar:
            acoustic_model_option = st.selectbox("Which acoustic model?",
                                                 ('Model 50k', 'Model 130k'))
            st.info(f"You selected:{acoustic_model_option}")
            if acoustic_model_option == "Model 50k":
                st.session_state.acoustic_model = "model1"
            elif acoustic_model_option == "Model 130k":
                st.session_state.acoustic_model = "model2"
            st.session_state.lm_option = st.selectbox("Which language model?",
                                                      ("CTC", "CTC + 4-gram"))
            st.info(f"You selected:{st.session_state.lm_option}")

            if st.session_state.acoustic_model == "" or st.session_state.lm_option == "":
                st.sidebar.warning("Setting must not be empty")

            if st.session_state.predict_arg != {"model": st.session_state.acoustic_model,
                                                "lm": st.session_state.lm_option}:
                st.session_state.predict_arg = {"model": st.session_state.acoustic_model,
                                                "lm": st.session_state.lm_option}
                arg_request = json.dumps(st.session_state.predict_arg)
                response = requests.post(f"{URL}/pattern", data=arg_request)
                arg_return = response.json()

                return arg_return

    def upload_file(self, col):
        st.session_state.audio_file = col.file_uploader("Upload audio", type=['wav', 'mp3'])
        if st.session_state.audio_file is not None:
            col.success("File uploaded successfully")
            col.audio(st.session_state.audio_file)
            st.session_state.upload_complete = True
        else:
            col.info("Please upload a audio file")
            st.session_state.upload_complete = False

        return st.session_state.upload_complete, st.session_state.audio_file

    def display_app_header(self, main_txt, is_sidebar=False):
        """
        function to display major headers at user interface
        ----------
        main_txt: str -> the major text to be displayed
        sub_txt: str -> the minor text to be displayed 
        is_sidebar: bool -> check if its side panel or major panel
        """

        html_temp = f"""
        <h2 style = "color:#F74369; text_align:center; font-weight: bold;"> {main_txt} </h2>
        </div>
        """
        if is_sidebar:
            st.sidebar.markdown(html_temp, unsafe_allow_html=True)
        else:
            st.markdown(html_temp, unsafe_allow_html=True)
