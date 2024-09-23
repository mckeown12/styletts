from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import numpy as np
import scipy.io.wavfile as wavfile
from styletts2importable import inference, compute_style
from txtsplit import txtsplit
from tqdm import tqdm
from typing import Optional
import datetime
import os
import tempfile
from enum import Enum

# Allow all origins
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceChoice(str, Enum):
    female1 = "f-us-1"
    female2 = "f-us-2"
    female3 = "f-us-3"
    female4 = "f-us-4"
    male1 = "m-us-1"
    male2 = "m-us-2"
    male3 = "m-us-3"
    male4 = "m-us-4"

# Precompute the styles for predefined voices
voice_styles = {}
for voice in VoiceChoice:
    voice_path = f'voices/{voice.value}.wav'
    voice_styles[voice.value] = compute_style(voice_path)

default_ref = voice_styles[VoiceChoice.male4.value]  # Default reference voice

def synthesize(text, voice, lngsteps=4):
    if text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 50000:
        raise HTTPException(status_code=400, detail="Text must be less than 50,000 characters")
    print("*** saying ***")
    print(text)
    print("*** end ***")
    texts = txtsplit(text)
    audios = []
    for t in tqdm(texts):
        print(t)
        audios.append(inference(t, voice, alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1))
    return (24000, np.concatenate(audios))

@app.post("/text-to-speech")
async def text_to_speech(
    text: str = Form(...),
    referenceWavFile: Optional[UploadFile] = File(None),
    voice_choice: Optional[VoiceChoice] = Form(None)
):
    if referenceWavFile is not None:
        print("Using uploaded reference WAV file")
        # Validate the file extension
        if not referenceWavFile.filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Invalid file format. Only WAV files are allowed.")
        
        # Save the uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(await referenceWavFile.read())
        temp_file.close()
        # Compute the style from the temporary file
        ref = compute_style(temp_file.name)
        print(ref)
        # Clean up the temporary file
        os.remove(temp_file.name)
    elif voice_choice is not None:
        print(f"Using selected voice: {voice_choice}")
        ref = voice_styles[voice_choice.value]
    else:
        print("Using default reference voice")
        ref = default_ref

    content = synthesize(text, ref)
    filename = f"audio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    
    wavfile.write(filename, content[0], (content[1] * 32767).astype(np.int16))
    
    return FileResponse(filename, media_type="audio/wav", filename=filename)

@app.on_event("shutdown")
def cleanup():
    # Clean up temporary files
    for file in os.listdir():
        if file.startswith("audio_") and file.endswith(".wav"):
            os.remove(file)
