import base64
import json

import os

import bark_tts
import coqui_tts

from fastapi import FastAPI, Header, Query, File, UploadFile, HTTPException
from typing import Optional, Union, Dict, Annotated
from fastapi.responses import Response, FileResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_auth_token(auth_token: str) -> bool:
    valid_tokens = os.getenv("AUTH_TOKEN", "").split(",")
    valid_tokens = [token.strip() for token in valid_tokens]  # Remove leading and trailing spaces
    return auth_token in valid_tokens


@app.post("/bark_wav_history")
def generate_suno(text_prompt: str, history_prompt: Optional[str] = None, seed: Optional[int] = -1,
                  x_auth_token: Annotated[str | None, Header()] = None):
    if not validate_auth_token(x_auth_token):
        return HTTPException(status_code=401, detail="Invalid x_auth_token")

    audio, history_prompt_data, used_seed = bark_tts._generate_suno_data(text_prompt, history_prompt, seed)

    if audio is None:
        return None

    wav_base64 = base64.b64encode(audio).decode('utf-8')
    npz_base64 = base64.b64encode(history_prompt_data).decode('utf-8')
    return Response(content=json.dumps({
        'wav_file': wav_base64,
        'npz_file': npz_base64,
        'seed': used_seed,
    }), media_type="application/json")


@app.post("/bark_wav")
def generate_suno_wav(text_prompt: str, history_prompt: Optional[str] = None, seed: Optional[int] = -1,
                      x_auth_token: Annotated[str | None, Header()] = None):
    if not validate_auth_token(x_auth_token):
        return HTTPException(status_code=401, detail="Invalid x_auth_token")

    audio, _, used_seed = bark_tts._generate_suno_data(text_prompt, history_prompt, seed)

    if audio is None:
        return None
        
    headers = {'Content-Disposition': 'attachment; filename="bark_audio.wav"'}
    return Response(content=audio, headers=headers, media_type="audio/wav")


@app.post("/xtts_wav")
def xtts_wav(text: str,
             language: str = Query("eu", enum=['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'jp', 'hu', 'ko', 'ja']),
             replace_abbreviations: bool = False,
             clone_speaker_wav: UploadFile = File(None),
             x_auth_token: Annotated[str | None, Header()] = None,
             ):
    if not validate_auth_token(x_auth_token):
        return HTTPException(status_code=401, detail="Invalid x_auth_token")

    audio = coqui_tts.tts_generation_xtts(
                                     text=text,
                                     language=language,
                                     replace_abbreviations=replace_abbreviations,
                                     clone_wav=clone_speaker_wav,
                                     )

    if audio is None:
        return None

    headers = {'Content-Disposition': 'attachment; filename="xtts.wav"'}
    return Response(content=audio, headers=headers, media_type="audio/wav")


#@app.post("/tts")
#def tts_wav(text: str,
#            model: str = Query("tts_models/multilingual/multi-dataset/xtts_v1.1",
#                               description=f"Available models: {', '.join(coqui_tts.list_models())}"),
#            speaker: str = Query("", description=f"Available speakers: {', '.join(coqui_tts.list_speakers())}"),
#            language: str = Query("", description=f"Available speakers: {', '.join(coqui_tts.list_languages())}"),
#            replace_abbreviations: bool = False,
#            clone_speaker_wav: UploadFile = File(None),
#            x_auth_token: Annotated[str | None, Header()] = None,
#            ):
#    if not validate_auth_token(x_auth_token):
#        return HTTPException(status_code=401, detail="Invalid x_auth_token")
#
#    print("clone_speaker_wav")
#    print(clone_speaker_wav)
#    print("language")
#    print(language)
#    print("speaker")
#    print(speaker)
#
#    audio = coqui_tts.tts_generation(model,
#                                     text=text,
#                                     speaker=speaker,
#                                     language=language,
#                                     replace_abbreviations=replace_abbreviations,
#                                     emotion=None,
#                                     speed=1.0,
#                                     voice_dir=None,
#                                     wav_sample_rate=None,
#                                     clone_wav=clone_speaker_wav,
#                                     )
#
#    if audio is None:
#        return None
#
#    return Response(content=audio, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
