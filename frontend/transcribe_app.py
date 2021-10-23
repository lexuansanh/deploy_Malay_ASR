import streamlit as st
import wave
import os
import time
import base64
import requests

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")

URL = "http://0.0.0.0:8080"

if "file" not in st.session_state:
    st.session_state.file = {"file_name": ""}
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "submit" not in st.session_state:
    st.session_state.submit = 0
if "noise" not in st.session_state:
    st.session_state.noise = False
if "noise2" not in st.session_state:
    st.session_state.noise2 = False
if "noise3" not in st.session_state:
    st.session_state.noise3 = False
if "duration_scope" not in st.session_state:
    st.session_state.duration_scope = 240
if "augmentation" not in st.session_state:
    st.session_state.augmentation = {"duration_scope": 240, "filter": [False, False, False]}

###############################
# Config
###############################
st.set_page_config(
    page_title="Malay Speech To Text",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .reportview-container {
        background-image: linear-gradient(to right,#faa3ff, #ffffff);color:#ffffff;)
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
div.stDownloadButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stDownloadButton > button:hover {
    background-color: #ffffff;
    color:#f00000;
    }
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #ffffff;
    color:#f00000;
    }
</style>""", unsafe_allow_html=True)

SAMPLE_RATE = 16_000
AUDIO_EXTENSION = ".wav"
TEMP_DIR = "../temp_dir"


###############################
# Packages
###############################


################################
# Functions
################################
# Background

def display_app_header(main_txt, sub_txt, is_sidebar=False):
    """
    function to display major headers at user interface
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <h2 style = "color:#F74369; text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#BB1D3F; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    else:
        st.markdown(html_temp, unsafe_allow_html=True)


################################
# GUI
################################
# Title
header_text = "Speech To Text for Malaysia Speech"
st.title(header_text)
main_txt = "Speech To Text for Malaysia Speech"
sub_txt = "By NamND40 & SanhLX"
display_app_header(main_txt, sub_txt, is_sidebar=True)

tool_sidebar = st.sidebar.selectbox(
    "Tools",
    ("Audio Transcription", "Realtime Recording")
)
###############################
# Sidebar


if tool_sidebar == "Audio Transcription":
    with st.form(key="main_form"):
        with st.sidebar:
            with st.expander("Settings"):
                with st.form("info_transcribe_form"):
                    st.markdown("**Transcription Information**")
                    st.session_state.duration_scope = st.slider("Max Duration", min_value=1, max_value=240)
                    st.markdown("**Filter**")
                    st.session_state.noise = st.checkbox("Noise")
                    st.session_state.noise2 = st.checkbox("Noise2")
                    st.session_state.noise3 = st.checkbox("Noise3")
                    user_word = st.text_input("Enter a keyword", "...")
                    select_language = st.radio('Language', ('All', 'English', 'Malaysia'))
                    include_retweets = st.checkbox('Include retweets in data')
                    num_of_tweets = st.number_input('Maximum number of tweets', 100)

                    sidebar_submitted = st.form_submit_button("Done")
                    if sidebar_submitted:
                        st.write(f"Duration scope: {st.session_state.duration_scope}s")
                        st.write("Filter:", [st.session_state.noise, st.session_state.noise2, st.session_state.noise3])

            main_submitted = st.form_submit_button("Save")
            if main_submitted:
                st.session_state.augmentation = {"duration_scope": st.session_state.duration_scope, "filter": [st.session_state.noise, st.session_state.noise2, st.session_state.noise3]}
                st.success("Saved")

    ###############################
    # Main components
    audio_file = None
    data = {"word": ""}
    file = {}
    with st.expander("Get Audio", expanded=True):
        with st.form("info_form"):
            col1, col2 = st.columns((1, 1))
            st.session_state.audio_file = col1.file_uploader("Upload audio", type=['wav', 'mp3'])
            col2.markdown("Audio Information")
            _submitted = st.form_submit_button("OK")
            if _submitted:
                st.session_state.submit = 1
            if st.session_state.audio_file is not None and st.session_state.submit == 1:
                    # time.sleep(0.5)
                file_details = [f"Filename: {st.session_state.audio_file.name}",
                                f"FileType: {st.session_state.audio_file.type}",
                                f"FileSize: {st.session_state.audio_file.size} bytes"]
                for file_de in file_details:
                    col2.info(file_de)
                col1.audio(st.session_state.audio_file)
                values = {"file": (st.session_state.audio_file.name, st.session_state.audio_file, "audio/wav")}
                response = requests.post(f"{URL}/upfile", files=values)
                st.session_state.file = response.json()

    trans_btn = st.button("Transcribe")
    if trans_btn and st.session_state.file["file_name"] != "":
        response = requests.post(f"{URL}/predict", json=st.session_state.file)
        data = response.json()

        # st.markdown('**Processing**...')
        # my_bar = st.progress(0)
        # #package stuff to send and perform POST request
        # for percent in range(100):
        #     time.sleep(0.1)
        #     my_bar.progress(percent + 1)

    transcript_result = st.text_area("Text Transcript", value=data["word"], height=400)

    if transcript_result is not None:
        data_down = transcript_result.strip()

        current_time = time.strftime("%H%M%S-%d%M%y")
        file_name = "transcript_" + str(current_time)
        st.download_button(label="Save transcript",
                           data=data_down,
                           file_name=f'{file_name}.txt',
                           mime='text/csv')


def record(stt_button):
    stt_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if ( value != "") {
                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
            }
        }
        recognition.start();
        """))

    result = streamlit_bokeh_events(
        stt_button,
        events="GET_TEXT",
        key="listen",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)

    if result:
        if "GET_TEXT" in result:
            st.write(result.get("GET_TEXT"))


if tool_sidebar == "Realtime Recording":
    record_btn = Button(label="Record", width=100)
    transcript_result = st.text_area("Text Transcript", height=400)
    if record_btn:
        record(record_btn)

    if transcript_result is not None:
        data_down = transcript_result.strip()

        current_time = time.strftime("%H%M%S-%d%M%y")
        file_name = "transcript_" + str(current_time)
        st.download_button(label="Save transcript",
                           data=data_down,
                           file_name=f'{file_name}.txt',
                           mime='text/csv')

###############################


################################
# st.sidebar.subheader("About App")
# st.sidebar.text("Speech To Text Based App with Streamlit")
# st.sidebar.info("...")


# st.sidebar.subheader("By")
# st.sidebar.text("NamND40")
# st.sidebar.text("SanhLX")
################################
