# download punkt model
import nltk
nltk.download('punkt')

# download xtts model
import coqui_tts
model_name = "tts_models/multilingual/multi-dataset/xtts_v1.1"
tts_model = coqui_tts.load_model(model_name, "", False)
