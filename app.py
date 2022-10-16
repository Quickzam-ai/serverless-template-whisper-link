import torch
import whisper
import os
import base64
from io import BytesIO
from pytube import YouTube 

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("large")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    #Parse out your arguments
#     mp3BytesString = model_inputs.get('mp3BytesString', None)
#     if mp3BytesString == None:
#         return {'message': "No input provided"}
    
#     mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
#     with open('input.mp3','wb') as file:
#         file.write(mp3Bytes.getbuffer())

    
    # Run the model
    yt = YouTube(model_inputs.get("myByteString"))
    audio = whisper.load_audio(yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download())
    translate_options = dict(task="translate")
    result = model.transcribe(audio, **translate_options)
    output = {"text":result["text"]}
    #os.remove("input.mp3")
    # Return the results as a dictionary
    return output
