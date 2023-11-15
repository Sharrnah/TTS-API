import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"
# os.environ["SUNO_USE_SMALL_MODELS"] = str(False)

import io
import numpy as np
import torch
import torchaudio
import random

torch.jit.enable_onednn_fusion(True)

from bark.generation import (
    generate_text_semantic,
    preload_models,
    generate_coarse,
    generate_fine
)
from bark import SAMPLE_RATE

from vocos import Vocos

from scipy.io.wavfile import write as write_wav
from typing import Optional, Union, Dict

import audio_tools

# make pytorch fully deterministic (disabling CuDNN benchmarking can slow down computations)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

GEN_TEMP = 0.6
WAV_TEMP = 0.7
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

sample_rate_vocos = 24000

USE_GPU = True

DEVICE = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
print("Device Using:")
print(DEVICE)

vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(DEVICE)

with torch.no_grad():
    preload_models(
        text_use_gpu=USE_GPU,
        coarse_use_gpu=USE_GPU,
        # coarse_use_small=True,
        fine_use_gpu=USE_GPU,
        codec_use_gpu=USE_GPU,
    )


@torch.no_grad()
def apply_vocos_on_audio(audio_data, sample_rate=24000):
    global DEVICE
    if isinstance(audio_data, bytes):
        audio_data = io.BytesIO(audio_data)

    y, sr = torchaudio.load(audio_data)
    if y.size(0) > 1:  # mix to mono
        y = y.mean(dim=0, keepdim=True)
    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=sample_rate)
    y = y.to(DEVICE)
    bandwidth_id = torch.tensor([2]).to(DEVICE)  # 6 kbps
    y_hat = vocos(y, bandwidth_id=bandwidth_id)

    audio_data_np_array = audio_tools.resample_audio(y_hat, sample_rate, sample_rate, target_channels=-1,
                                                     is_mono=True, dtype="float32")

    audio_data_np_array = np.int16(audio_data_np_array * 32767)  # Convert to 16-bit PCM

    buff = io.BytesIO()
    write_wav(buff, sample_rate, audio_data_np_array)

    buff.seek(0)
    return buff


@torch.no_grad()
def semantic_to_audio_tokens(
        semantic_tokens: np.ndarray,
        history_prompt: Optional[Union[Dict, str]] = None,
        temp: float = 0.7,
        silent: bool = False,
        output_full: bool = False,
):
    print("Generating coarse tokens...")
    coarse_tokens = generate_coarse(
        semantic_tokens, history_prompt=history_prompt, temp=temp, silent=silent, use_kv_caching=True
    )
    print("Generating fine tokens...")
    fine_tokens = generate_fine(coarse_tokens, history_prompt=history_prompt, temp=0.5)

    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation
    return fine_tokens


def get_history_prompt_file(full_generation):
    assert (isinstance(full_generation, dict))
    assert ("semantic_prompt" in full_generation)
    assert ("coarse_prompt" in full_generation)
    assert ("fine_prompt" in full_generation)
    buffer = io.BytesIO()
    np.savez(buffer, **full_generation)
    buffer.seek(0)
    return buffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # if is_torch_available():
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    # if is_torch_npu_available():
    #    torch.npu.manual_seed_all(seed)
    # if is_torch_xpu_available():
    #    torch.xpu.manual_seed_all(seed)


@torch.no_grad()
def _generate_suno_data(text_prompt: str, history_prompt: Optional[str] = None, seed: Optional[int] = -1):
    global DEVICE, GEN_TEMP, WAV_TEMP
    """
    Generate audio based on the given text prompt and optional history prompt.
    If history prompt is provided, it is used along with the text prompt.
    The generated audio is saved as a WAV file and returned as a response.
    """
    worker_seed = seed
    if seed is None or seed == -1:
        worker_seed = random.randint(0, 2 ** 32 - 1)
    set_seed(worker_seed)

    print("generating text semantics..")
    if history_prompt is None or history_prompt == "":
        history_prompt = None
        # audio_array = generate_audio(text_prompt)
        semantic_tokens = generate_text_semantic(
            text_prompt,
            temp=GEN_TEMP,
            min_eos_p=0.05,  # this controls how likely the generation is to end
            use_kv_caching=True,
        )
    else:
        # audio_array = generate_audio(text_prompt, history_prompt)
        semantic_tokens = generate_text_semantic(
            text_prompt,
            history_prompt=history_prompt,
            temp=GEN_TEMP,
            min_eos_p=0.05,  # this controls how likely the generation is to end
            use_kv_caching=True,
        )

    print("generating audio tokens..")
    history_prompt_data = semantic_to_audio_tokens(
        semantic_tokens, history_prompt=history_prompt, temp=WAV_TEMP, silent=False,
        output_full=True,
    )
    npz_file = get_history_prompt_file(full_generation=history_prompt_data)

    audio_tokens_torch = torch.from_numpy(history_prompt_data["fine_prompt"]).to(DEVICE)
    print("vocos reconstruction from audio tokens..")
    features = vocos.codes_to_features(audio_tokens_torch)
    audio_array = vocos.decode(features, bandwidth_id=torch.tensor([2], device=DEVICE)).cpu().numpy()

    audio_array = audio_tools.resample_audio(audio_array, 24000, 44100, target_channels=-1,
                                             is_mono=True, dtype="float32")
    print("audio cleanup..")
    audio_array = audio_tools.audio_cleanup(audio_array, sample_rate=44100, skip_infinity_lufs=True)
    audio_array = np.int16(audio_array * 32767)  # Convert to 16-bit PCM

    # Save the audio as a WAV file
    # audio_file = "output.wav"
    buff = io.BytesIO()
    write_wav(buff, 44100, audio_array)
    buff.seek(0)

    # apply vocos additionally on final audio
    # print("vocos on final audio..")
    # buff = apply_vocos_on_audio(buff, 44100)

    print("audio generated.")

    return buff.getvalue(), npz_file.getvalue(), worker_seed
