FROM python:3.9

LABEL maintainer="sanhlx, namnd"
LABEL repository="wav2vec2-xlsr-53"

# Set the working directory to /app
WORKDIR /app

# Install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

RUN python -m pip install --upgrade pip
COPY requirements.txt .
# Install python packages
RUN pip install -r requirements.txt

#install ctcdecoder
RUN git clone --recursive https://github.com/parlance/ctcdecode.git /tmp/ctcdecode \
 && cd /tmp/ctcdecode && pip3 install .

COPY . .

# Start server using config file
EXPOSE 8080

CMD ["python", "server.py"]