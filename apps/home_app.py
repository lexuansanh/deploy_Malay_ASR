import os
import streamlit as st
from hydralit import HydraHeadApp

MENU_LAYOUT = [1,1,1,7,2]


class HomeApp(HydraHeadApp):
    
    def __init__(self,
                 title="Malay Speech",
                 **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        try:
            # set title of home page
            st.markdown("""<h1 style='text-align:center;
                        padding: 0px 0px;
                        color:black;
                        font-size:200%;
                        '>Speech to text for Malaysia language</h1>""",
                        unsafe_allow_html=True)     
            
            st.markdown("""<h2 style='text-align: center;'>
                        This model used in tool is based on paper <a href=http:////arxiv.org/pdf/2006.11477.pdf>Wav2vec2</a>.""",
                        unsafe_allow_html=True)

            _, acoustic_col, lm_col,  kenlm_col, _ = st.columns([4,1,1,1,4])
            
            if acoustic_col.button('Wav2vec2'):
                self.do_redirect("https://arxiv.org/pdf/2006.11477.pdf")
                
            if lm_col.button('4-gram'):
                self.do_redirect("https://web.stanford.edu/~jurafsky/slp3/3.pdf")
                
            if kenlm_col.button('KenLM'):
                self.do_redirect("https://github.com/kpu/kenlm")                

            # set information for malay speech app
            _,_,col_logo, col_text,_ = st.columns(MENU_LAYOUT)
            col_logo.image(os.path.join("./", "resources","data.png"), width=80,)
            col_text.subheader("This toolkit has multiple applications, "
                               "each application could be run individually, "
                               "specific app is below:")

            st.markdown('<br><br>',unsafe_allow_html=True)
            
            _,_,col_logo, col_text,col_btn = st.columns(MENU_LAYOUT)
            col_logo.image(os.path.join(".","resources", "audio_to_text.png"),width=50,)
            col_text.info("Speech to text: transcribe audio speech to text")
            
            _,_,col_logo, col_text,col_btn = st.columns(MENU_LAYOUT)
            col_logo.image(os.path.join(".","resources", "record_to_text.jpg"),width=50,)
            col_text.info("Realtime ASR: record audio and transcribe it to text right away")

            _,_,col_logo, col_text,col_btn = st.columns(MENU_LAYOUT)
            col_logo.image(os.path.join(".","resources", "enhance.png"),width=50,)
            col_text.info("Enhancement Audio App: Nosie Reduction, Speech Enhancement, Super Resolutions")

            _,_,col_logo, col_text,col_btn = st.columns(MENU_LAYOUT)
            col_logo.image(os.path.join(".","resources", "diarization.png"),width=50,)
            col_text.info("Speaker Diarization App: Speaker Change Detection, Speaker Diarization")

            _,_,col_logo, col_text,col_btn = st.columns(MENU_LAYOUT)
            col_logo.image(os.path.join(".","resources", "class.png"),width=50,)
            col_text.info("Classification app: age, emotion, gender, speaker overlap, realtime")

            
        except Exception as e:
            st.image(os.path.join(".","resources","failure.png"),width=100,)
            st.error('An error has occurred, someone will be punished for your inconvenience, we humbly request you try again.')
            st.error('Error details: {}'.format(e))
        
        