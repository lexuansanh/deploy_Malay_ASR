import os
import time
#import av
#import cv2
import streamlit as st
from hydralit import HydraHeadApp
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


TEMP_DIR = "./temp_dir"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


class RealtimeAsrApp(HydraHeadApp):
    
    def __init__(self, title, **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        
        try:
            st.title('Realtime ASR')
            st.subheader('App to record audio voice realtime and transcribe it to text right away')
            st.markdown('<br><br>',unsafe_allow_html=True)
            
            self.display_app_header(self.title, True)
            
            record_options = st.sidebar.selectbox("What recording options?", 
                                        ('With video', 'Only Audio'))
            
            st.sidebar.info(f'You selected: {record_options}')
            
            sample_rate_options = st.sidebar.selectbox("Which sampling frequency to record?", 
                                        ('44100 Hz', '16000 Hz'))
            
            st.sidebar.info(f'You selected: {sample_rate_options}')
            
            _, col2, _ = st.columns((1,8,1))
            
            # text_output = col2.text_area("Text Transcript", height=300)
            if "Only Audio" in record_options:
                recorder = Record(is_audio=True, is_video=False)
                recorder.run()
            else:
                recorder = Record(is_audio=True, is_video=True)
                recorder.run()
                

            # # Enable record function
            # if text_output is not None:
            #     data_down = text_output.strip()
            
            #     current_time = time.strftime("%H%M%S-%d%M%y")
            #     file_name = "transcript_" + str(current_time)
            #     col2.download_button(label="Save transcript",
            #                         data=data_down,
            #                         file_name=f'{file_name}.txt',
            #                         mime='text/csv')
        
        except Exception as e:
            st.image(os.path.join(".","resources","failure.png"),width=100,)
            st.error('An error has occurred, someone will be punished for your inconvenience, we humbly request you try again.')
            st.error('Error details: {}'.format(e))
            
    def display_app_header(self, main_txt, is_sidebar = False):
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
            st.sidebar.markdown(html_temp, unsafe_allow_html = True)
        else: 
            st.markdown(html_temp, unsafe_allow_html = True)       
            
class Record:
    
    def __init__(self, is_audio=True, is_video=False, **kwargs):
        self.__dict__.update(kwargs)
        self.is_audio = is_audio
        self.is_video = is_video
        
        current_time = time.strftime("%H%M%S-%d%M%y")
        file_name = "audio_" + str(current_time)
        self.output_path = os.path.join(TEMP_DIR, file_name)
    
    def out_recorder_factory(self) -> MediaRecorder:
        if self.is_video:
            output_extension = "flv"
        else:
            output_extension = "wav"
        return MediaRecorder(
            f"{self.output_path}.{output_extension}", format=output_extension
        )
    
    def run(self):
        webrtc_streamer(
            key="loopback",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={
                "video": self.is_video,
                "audio": self.is_audio,
            },
            out_recorder_factory=self.out_recorder_factory,
        )