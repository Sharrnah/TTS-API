import os

# download punkt model
import nltk
nltk.download('punkt')

# download xtts model
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS(model_name=model_name, progress_bar=True, gpu=False)
