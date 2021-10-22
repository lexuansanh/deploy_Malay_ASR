from re import I
import streamlit as st

# Text/ Title
st.title("Streamlit tutorial")

# Header/ Subheader
st.header("This is a header")
st.subheader("This is a subheader")

# Text
st.text("This is a text")

# Markdown
st.markdown("This is a markdown")

# Error/ Colorful Text
st.success("This is a success")

st.info("This is a info")

st.warning("This is a warning")

st.error("This is a error")

st.exception("This is a exception")

# Writing Text
st.write("This is a write")

st.write(range(10))

# Images
from PIL import Image

img = Image.open("examples/images/stt.png")
st.image(img, width=300, caption="Image")

# Videos
vid_file = open("examples/videos/stt.mp4", 'rb').read()
st.video(vid_file)

# Audio
audio_file = open("examples/audios/stt.wav", "rb").read()
st.audio(audio_file)

# Widgets
# checkbox
if st.checkbox("Show/Hide"):
    st.text("showing or hiding widget")

# Radio buttons
status = st.radio("What is your status", ('Active', 'Inactive'))

if status == "Active":
    st.success("you are active")
else:
    st.warning("you are inactive")

# Select boxes
occupation = st.selectbox("Your Occupation", ['Programer', 'DS'])
st.write("you selected this option", occupation)

# Multiselect
location = st.multiselect("English", ("hi", "hello", "goodbye"))
st.write("you selected", len(location), "locations")

# Slider
level = st.slider("What is your level", 1, 5)

# Buttons
st.button("Simple Buttons")

if st.button("About"):
    st.text("Streamlit is cool")

# Text Input
name = st.text_input("Enter your firstname", "Type here")
if st.button("Submit"):
    result = name.title()
    st.success(result)

# Text area
message = st.text_area("Enter your firstname", "Type here")
if st.button("submit"):
    result = message.title()
    st.success(result)

# Data input
import datetime
today = st.date_input("Today is", datetime.datetime.now())

# Time input
the_time = st.time_input("The time is", datetime.time())

# Display Json
st.text("Display Json")
st.json({'name': "Nam", "gender": "male"})

# Display raw code
with st.echo():
    import pandas as pd
    df = pd.DataFrame()

# Progress bar
import time
my_bar = st.progress(0)
for p in range(10):
    my_bar.progress(p + 1)

# Spinner
with st.spinner("Waiting... "):
    time.sleep(5)
st.success("Finished!")

# Balloons
# st.balloons()

# Sidebars
st.sidebar.header("About")
st.sidebar.text("this is streamlit tut")

# functions
@st.cache
def run_fxn():
    return range(100)

st.write(run_fxn())

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3])
st.pyplot(fig)

# DataFrame
# st.dataframe(df)

# Tables
# st.table(df)

with st.form("my_form"):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)

st.write("Outside the form")