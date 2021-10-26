import os
from hydralit import HydraApp
import hydralit_components as hc
import streamlit as st
from PIL import Image
import apps

#----------------------------------------------------------------
# Set config for page
im = Image.open("./resources/favicon.png")
st.set_page_config(page_title='Malay Speech',
                   page_icon=im,
                   layout='wide',
                   initial_sidebar_state='auto',)

# Background image
st.markdown(
    """
    <style>
    .reportview-container {
        background-image: linear-gradient(to right,#ffd0d0, #ffd0d0);color:#000000;)
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Button color
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

#----------------------------------------------------------------
if __name__ == "__main__":
    over_theme = {'txc_inactive': '#000000'}
    #this is the host application, we add children to it and that's it!
    app = HydraApp(
        title='Malay Speech',
        favicon=im,
        #add a nice banner, this banner has been defined as 5 sections with spacing defined by the banner_spacing array below.
        use_banner_images=["./resources/micro.png",None,{'header':"<h1 style='text-align:center;padding: 0px 0px;color:black;font-size:200%;'>Malay Speech</h1><br>"},None,"./resources/lock.png"], 
        banner_spacing=[5,30,60,20,4],
        navbar_theme=over_theme,
        navbar_animation=True,
        # navbar_sticky=True
    )
    
    #Home button will be in the middle of the nav list now
    app.add_app("Home", icon="🏠", app=apps.HomeApp(title='Home'),is_home=True)
    app.add_app("Speech to text", icon="🗣️", app=apps.SpeechToTextApp(title='Speech to text'))
    app.add_app("Realtime ASR", icon="🎙️", app=apps.RealtimeAsrApp(title='Realtime Asr'))
    app.add_app("Enhancement Audio", icon="🎚️", app=apps.HomeApp(title='Enhancement Audio'))
    app.add_app("Speaker Diarization", icon="📼", app=apps.HomeApp(title='Speaker Diarization'))
    app.add_app("Classification", icon="🎛️", app=apps.HomeApp(title='Classification'))
    app.add_app("About", icon="🏴󠁭󠁥󠀲󠀲󠁿", app=apps.HomeApp(title='About'))
    
    # Signup and sign in
    # app.add_app("Signup", icon="🛰️", app=apps.SignUpApp(title='Signup'), is_unsecure=True)
    # app.add_app("Login", apps.LoginApp(title='Login'),is_login=True) 
    
    # # custom loading app for a custom transition between apps
    # app.add_loader_app(apps.MyLoadingApp(delay=0))
    
    # # login with guest
    # app.enable_guest_access()
    
    app.run()
