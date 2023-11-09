import io
import wave

import numpy as np
import torch
import pyloudnorm
import torchaudio


def _resample(smp, scale=1.0):
    """Resample a sound to be a different length

    Sample must be mono.  May take some time for longer sounds
    sampled at 44100 Hz.

    Keyword arguments:
    scale - scale factor for length of sound (2.0 means double length)

    """
    # f*ing cool, numpy can do this with one command
    # calculate new length of sample
    n = round(len(smp) * scale)
    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    return np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False), # where to interpret
        np.linspace(0.0, 1.0, len(smp), endpoint=False), # known positions
        smp, # known data points
    )


def _interleave(left, right):
    """Given two separate arrays, return a new interleaved array

    This function is useful for converting separate left/right audio
    streams into one stereo audio stream.  Input arrays and returned
    array are Numpy arrays.

    See also: uninterleave()

    """
    return np.ravel(np.vstack((left, right)), order='F')


def _uninterleave(data):
    """Given a stereo array, return separate left and right streams

    This function converts one array representing interleaved left and
    right audio streams into separate left and right arrays.  The return
    value is a list of length two.  Input array and output arrays are all
    Numpy arrays.

    See also: interleave()

    """
    return data.reshape(2, len(data)//2, order='F')


def resample_wav_simple(wav, sr, new_sr):
    wav = wav.unsqueeze(0)
    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sr)
    wav = transform(wav)
    return wav.squeeze(0)


def resample_audio(audio_chunk, recorded_sample_rate, target_sample_rate, target_channels=-1, is_mono=None, dtype="int16"):
    """
    Resample audio data and optionally convert between stereo and mono.

    :param audio_chunk: The raw audio data chunk as bytes, NumPy array or PyTorch Tensor.
    :param recorded_sample_rate: The sample rate of the input audio.
    :param target_sample_rate: The desired target sample rate for the output.
    :param target_channels: The desired number of channels in the output.
        - '-1': Average the left and right channels to create mono audio. (default)
        - '0': Extract the first channel (left channel) data.
        - '1': Extract the second channel (right channel) data.
        - '2': Keep stereo channels (or copy the mono channel to both channels if is_mono is True).
    :param is_mono: Specify whether the input audio is mono. If None, it will be determined from the shape of the audio data.
    :param dtype: The desired data type of the output audio, either "int16" or "float32".
    :return: A NumPy array containing the resampled audio data.
    """
    # Determine the data type for audio data
    audio_data_dtype = np.int16 if dtype == "int16" else np.float32

    # Convert the audio chunk to a numpy array
    if isinstance(audio_chunk, torch.Tensor):
        audio_chunk = audio_chunk.detach().cpu().numpy()

    audio_data = np.frombuffer(audio_chunk, dtype=audio_data_dtype)

    # Determine if the audio is mono or stereo; assume mono if the shape has one dimension
    if is_mono is None:
        is_mono = len(audio_data.shape) == 1

    # If stereo, reshape the data to have two columns (left and right channels)
    if not is_mono:
        audio_data = audio_data.reshape(-1, 2)

    # Handle channel conversion based on the target_channels parameter
    # -1 means converting stereo to mono by taking the mean of both channels
    # 0 or 1 means selecting one of the stereo channels
    # 2 means duplicating the mono channel to make it stereo
    if target_channels == -1 and not is_mono:
        audio_data = audio_data.mean(axis=1)
    elif target_channels in [0, 1] and not is_mono:
        audio_data = audio_data[:, target_channels]
    elif target_channels == 2 and is_mono:
        audio_data = _interleave(audio_data, audio_data)

    # Calculate the scaling factor for resampling
    scale = target_sample_rate / recorded_sample_rate

    # Perform resampling based on whether the audio is mono or stereo
    # If mono or selected one channel, use _resample directly
    # If stereo, split into left and right, resample separately, then interleave
    if is_mono or target_channels in [0, 1, -1]:
        audio_data = _resample(audio_data, scale)
    else:  # Stereo
        left, right = _uninterleave(audio_data)
        left_resampled = _resample(left, scale)
        right_resampled = _resample(right, scale)
        audio_data = _interleave(left_resampled, right_resampled)

    # Return the resampled audio data with the specified dtype
    return np.asarray(audio_data, dtype=audio_data_dtype)


# Function to calculate LUFS
def calculate_lufs(audio, sample_rate):
    meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    return loudness


# Function to normalize the audio based on LUFS
def normalize_audio_lufs(audio, sample_rate, lower_threshold=-24.0, upper_threshold=-16.0, gain_factor=2.0):
    lufs = calculate_lufs(audio, sample_rate)

    print(f"LUFS: {lufs}")

    # If LUFS is lower than the lower threshold, increase volume
    if lufs < lower_threshold:
        print(f"audio is too quiet, increasing volume")
        gain = (lower_threshold - lufs) / gain_factor
        audio = audio * np.power(10.0, gain/20.0)

    # If LUFS is higher than the upper threshold, decrease volume
    elif lufs > upper_threshold:
        print(f"audio is too loud, decreasing volume")
        gain = (upper_threshold - lufs) * gain_factor
        audio = audio * np.power(10.0, gain/20.0)

    # Limit audio values to [-1, 1] (this is important to avoid clipping when converting to 16-bit PCM)
    audio = np.clip(audio, -1, 1)

    return audio, lufs


def _trim_silence(audio, silence_threshold=0.01):
    # Compute absolute value of audio waveform
    audio_abs = np.abs(audio)

    # Find the first index where the absolute value of the waveform exceeds the threshold
    start_index = np.argmax(audio_abs > silence_threshold)

    # Reverse the audio waveform and do the same thing to find the end index
    end_index = len(audio) - np.argmax(audio_abs[::-1] > silence_threshold)

    # If start_index is not 0, some audio at the start has been trimmed
    if start_index > 0:
        print(f"Trimmed {start_index} samples from the start of the audio")

    # If end_index is not the length of the audio, some audio at the end has been trimmed
    if end_index < len(audio):
        print(f"Trimmed {len(audio) - end_index} samples from the end of the audio")

    # Return the trimmed audio
    return audio[start_index:end_index]


def _remove_silence_parts(audio, sample_rate, silence_threshold=0.01, max_silence_length=1.1, keep_silence_length=0.06):
    audio_abs = np.abs(audio)
    above_threshold = audio_abs > silence_threshold

    # Convert length parameters to number of samples
    max_silence_samples = int(max_silence_length * sample_rate)
    keep_silence_samples = int(keep_silence_length * sample_rate)

    last_silence_end = 0
    silence_start = None

    chunks = []

    for i, sample in enumerate(above_threshold):
        if not sample:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                silence_duration = i - silence_start
                if silence_duration > max_silence_samples:
                    # Subtract keep_silence_samples from the start and add it to the end
                    start = max(last_silence_end - keep_silence_samples, 0)
                    end = min(silence_start + keep_silence_samples, len(audio))
                    chunks.append(audio[start:end])
                    last_silence_end = i
                silence_start = None

    # Append the final chunk of audio after the last silence
    if last_silence_end < len(audio):
        start = max(last_silence_end - keep_silence_samples, 0)
        end = len(audio)
        chunks.append(audio[start:end])

    if len(chunks) == 0:
        print("No non-silent sections found in audio.")
        return np.array([])
    else:
        print(f"found {len(chunks)} non-silent sections in audio")
        return np.concatenate(chunks)


def audio_cleanup(
        audio_data,
        skip_infinity_lufs=True,
        sample_rate=44100,

        normalize=True,
        normalize_lower_threshold=-24.0,
        normalize_upper_threshold=-16.0,
        normalize_gain_factor=1.35,

        trim_silence=True,

        remove_silence_parts=True,
        silence_threshold=0.03,
        keep_silence_length=0.20,
        max_silence_length=0.8
):
    # Normalize audio
    if normalize:
        audio_data, lufs = normalize_audio_lufs(audio_data, sample_rate, normalize_lower_threshold, normalize_upper_threshold, normalize_gain_factor)
        if lufs == float('-inf') and skip_infinity_lufs:
            print("Audio seems to be unusable. skipping")
            return None

    # Trim silence
    if trim_silence:
        audio_data = _trim_silence(audio_data)

    # Remove silence parts
    if remove_silence_parts:
        audio_data = _remove_silence_parts(audio_data, sample_rate, silence_threshold=silence_threshold,
                                               keep_silence_length=keep_silence_length,
                                               max_silence_length=max_silence_length)

    # return early if no audio data
    if len(audio_data) == 0:
        return None

    return audio_data


def numpy_array_to_wav_bytes(audio: np.ndarray, sample_rate: int = 22050) -> bytes:
    byte_stream = io.BytesIO()
    wavefile = wave.open(byte_stream, 'w')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)  # 2 bytes = 16-bit PCM
    wavefile.setframerate(sample_rate)
    audio = np.array(audio)
    wavefile.writeframes((audio * 32767).astype(np.int16).tobytes())
    wavefile.close()
    byte_stream.seek(0)
    return byte_stream.read()
