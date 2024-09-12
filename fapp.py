from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import scipy.io.wavfile as wavfile
from styletts2importable import inference, compute_style
from txtsplit import txtsplit
from tqdm import tqdm
from typing import Optional
import datetime

app = FastAPI()
ref = compute_style(f'voices/m-us-4.wav')

def synthesize(text, voice=ref, lngsteps=4):
    if text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 50000:
        raise HTTPException(status_code=400, detail="Text must be less than 50,000 characters")
    print("*** saying ***")
    print(text)
    print("*** end ***")
    texts = txtsplit(text)
    v = voice.lower()
    audios = []
    for t in tqdm(texts):
        print(t)
        audios.append(inference(t, voice, alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1))
    return (24000, np.concatenate(audios))

@app.post("/text-to-speech")
async def text_to_speech(text: str, referenceWavFile: Optional[bytes] = None):
    if referenceWavFile:
        # Save the uploaded file temporarily
        with open("temp_reference.wav", "wb") as f:
            f.write(referenceWavFile)
        ref = compute_style("temp_reference.wav")
    #else ref is global
    content = synthesize(text, ref)  # Assuming this returns a NumPy array
    
    # Generate a unique filename using datetime
    filename = f"audio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    
    # Assuming content is a 1D NumPy array of float32 values between -1 and 1
    wavfile.write(filename, 24000, (content * 32767).astype(np.int16))
    
    # Return the file as a downloadable response
    return FileResponse(filename, media_type="audio/wav", filename=filename)

@app.on_event("shutdown")
def cleanup():
    # Clean up temporary files
    import os
    for file in os.listdir():
        if file.startswith("audio_") and file.endswith(".wav"):
            os.remove(file)