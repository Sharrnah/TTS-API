import base64
import re
import os

os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS
import numpy as np
import torch
import audio_tools

import nltk

SAMPLE_RATE = 22050

model_sample_rate_mapping = {
    "tts_models/en/jenny/jenny": 48000,
    "tts_models/en/multi-dataset/tortoise-v2": 24000,
    "tts_models/multilingual/multi-dataset/your_tts": 16000,
    "tts_models/???/fairseq/vits": 16000,
    "tts_models/multilingual/multi-dataset/xtts_v2": 24000,
    "tts_models/multilingual/multi-dataset/xtts_v1.1": 24000,
    "tts_models/multilingual/multi-dataset/xtts_v1": 24000,

    "voice_conversion_models/multilingual/vctk/freevc24": 24000,
}

#tts_model = None
#model_name = ""
tts_vc = None
fairseq_lang = ""
use_gpu = False
voice_conversion_model = None


def get_sample_rate(model_name):
    if model_name in model_sample_rate_mapping:
        return model_sample_rate_mapping[model_name]
    return SAMPLE_RATE


def load_model(model_name, fairseq_lang, use_gpu):
    global SAMPLE_RATE

    SAMPLE_RATE = get_sample_rate(model_name)

    # replace ??? with the language code for fairseq models
    if model_name.startswith("tts_models/???/fairseq/vits") and fairseq_lang is not None and fairseq_lang != "":
        model_name = model_name.replace("???", fairseq_lang)

    # Init TTS with the target model name
    return TTS(model_name=model_name, progress_bar=True, gpu=use_gpu)


ABBREVIATIONS_EXCEPTIONS = ["CUDA", "NVIDIA", "RAM", "LAN", "ZIP", "PIN", "DOS", "WINDOWS", "NASA", "NATO", "COVID", "UNICEF", "NASCAR", "AIDS",
                            "ASAP", "AWOL",
                            "SWIFT", "CERN", "LASER", "SONAR", "RADAR", "SCUBA", "LIDAR"]


def split_abbreviations(text):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)

    new_sentences = []
    for sentence in sentences:
        # Tokenize the sentence
        tokens = nltk.word_tokenize(sentence)

        new_tokens = []
        for i, token in enumerate(tokens):
            # Check if the token is an abbreviation
            if (re.match(r"^[A-Z]{2,}$", token)
                    and not re.match(r".*[!?.]$", token)
                    and token not in ABBREVIATIONS_EXCEPTIONS):
                # Check if the previous and next tokens are also capitalized
                prev_token = tokens[i - 1] if i > 0 else ""
                next_token = tokens[i + 1] if i < len(tokens) - 1 else ""

                if re.match(r"^[A-Z]+$", prev_token) or re.match(r"^[A-Z]+$", next_token):
                    # If the surrounding tokens are capitalized, treat the current token as a part of a phrase
                    new_token = token
                else:
                    # Replace the abbreviation with the same letters separated by a period and a space
                    new_token = ". ".join(list(token)) + "."
            else:
                new_token = token

            new_tokens.append(new_token)

        # Join the tokens back together
        new_sentence = " ".join(new_tokens)
        new_sentences.append(new_sentence)

    # Join the sentences back together
    new_text = " ".join(new_sentences)

    return new_text


model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_model = load_model(model_name, "", use_gpu)


@torch.no_grad()
def tts_generation_xtts(text, language='en', replace_abbreviations=False, clone_wav=None):
    global SAMPLE_RATE
    global tts_model, tts_vc, use_gpu

    wav_data = None

    if language == "" or language is None:
        language = None

    if clone_wav == "" or clone_wav is None:
        clone_wav = None

    clone_wav_data = None

    if clone_wav is not None:
        clone_wav_data = clone_wav.file

    if replace_abbreviations:
        text = split_abbreviations(text)

    try:
        wav_data = tts_model.tts(text=text, language=language, speaker_wav=clone_wav_data)
    except Exception as e:
        print("Exception:", str(e))
        return

    SAMPLE_RATE = get_sample_rate(model_name)

    wav_bytes = audio_tools.numpy_array_to_wav_bytes(wav_data, SAMPLE_RATE)

    return wav_bytes


@torch.no_grad()
def tts_generation(arg_model_name, text, speaker=None, language=None, replace_abbreviations=False, emotion=None, speed=1.0, clone_wav=None, voice_dir=None, wav_sample_rate=None):
    global SAMPLE_RATE
    global tts_model, tts_vc, model_name, use_gpu

    wav_data = None

    if tts_model is None or arg_model_name != model_name:
        tts_model = load_model(model_name, language, use_gpu)

    if speaker == "" or speaker is None:
        speaker = None

    if language == "" or language is None:
        language = None

    if clone_wav == "" or clone_wav is None:
        clone_wav = None

    if voice_dir == "" or voice_dir is None:
        voice_dir = None

    try:
        # load voice conversion model if needed
        if tts_vc is None and clone_wav is not None and clone_wav != "" and not model_name.startswith("tts_models/multilingual/multi-dataset/xtts"):
            model_name = "voice_conversion_models/multilingual/vctk/freevc24"
            tts_vc = load_model(model_name, "", use_gpu)

        # TTS generation
        if (clone_wav is None or clone_wav == "" or model_name.startswith("tts_models/multilingual/multi-dataset/xtts")) and text is not None and text != "":
            print("normal tts generation")

            if replace_abbreviations:
                text = split_abbreviations(text)

            try:
                wav_data = tts_model.tts(text=text, speaker=speaker, language=language, emotion=emotion, speed=speed, speaker_wav=clone_wav, voice_dir=voice_dir)
            except Exception as e:
                print("Exception:", str(e))
                return

            SAMPLE_RATE = get_sample_rate(model_name)

        # TTS generation with voice conversion
        elif tts_vc is not None and clone_wav is not None and text is not None and text != "" and not model_name.startswith("tts_models/multilingual/multi-dataset/xtts"):
            print("tts with voice conversion")
            #wav_data = tts.tts_with_vc(
            #    text,
            #    language=language,
            #    speaker_wav=clone_wav,
            #)

            tts_sample_rate = get_sample_rate(model_name)

            SAMPLE_RATE = get_sample_rate("voice_conversion_models/multilingual/vctk/freevc24")

            try:
                wav_data = tts_model.tts(text=text, speaker=speaker, language=language, emotion=emotion, speed=speed, voice_dir=voice_dir)

                if tts_sample_rate != 16000:
                    wav_data = torch.FloatTensor(wav_data)  # convert list to torch tensor
                    wav_data = audio_tools.resample_wav_simple(wav_data, tts_sample_rate, 16000)

                wav_data = tts_vc.voice_conversion(source_wav=wav_data, target_wav=clone_wav)
            except Exception as e:
                print("Exception:", str(e))
                return

        # Voice conversion only
        elif clone_wav is not None and tts_vc is not None and (text is None or text == "") and wav_data is not None:
            print("voice conversion only")
            # decode wav_data from base64
            wav_data = base64.b64decode(wav_data)
            wav_data_numpy_array = np.frombuffer(wav_data, dtype=np.int16).copy()

            SAMPLE_RATE = get_sample_rate("voice_conversion_models/multilingual/vctk/freevc24")

            # 16000 is the freevc voice conversion input sample rate
            if wav_sample_rate is not None and wav_sample_rate != 16000:
                wav_data_numpy_array = audio_tools.resample_wav_simple(wav_data_numpy_array, wav_sample_rate, 16000)

            wav_data = tts_vc.voice_conversion(source_wav=wav_data_numpy_array, target_wav=clone_wav)

        wav_bytes = audio_tools.numpy_array_to_wav_bytes(wav_data, SAMPLE_RATE)

        return wav_bytes

    except Exception as e:
        print("Exception:", str(e))


def list_models():
    return TTS().list_models()


def list_speakers():
    global tts_model
    if tts_model is not None:
        return tts_model.speakers
    return []


def list_languages():
    global tts_model
    if tts_model is not None:
        return tts_model.languages
    return []
