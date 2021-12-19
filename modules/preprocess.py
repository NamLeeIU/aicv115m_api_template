from pydub import AudioSegment
import librosa
import pandas as pd
import numpy as np
import malaya_speech
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile as sf
import os


def zero_pad(path):
    pad_ms = 5000  # Add here the fix length you want (in milliseconds)
    audio = AudioSegment.from_wav(path)
    if pad_ms > len(audio):
        silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)
        padded = audio + silence  # Adding silence after the audio
        os.remove(path)
        padded.export(path, format='wav')


def pre_process(path, fname, dir_path):
    y, sr = librosa.load(path, res_type='kaiser_fast')
    y_ = librosa.effects.trim(y, top_db=20)[0]
    y_int = malaya_speech.astype.float_to_int(y)
    audio = AudioSegment(
        y_int.tobytes(),
        frame_rate=sr,
        sample_width=y_int.dtype.itemsize,
        channels=1
    )
    audio_chunks = split_on_silence(
        audio,
        min_silence_len=200,
        silence_thresh=-30,
        keep_silence=100,
    )
    y_ = sum(audio_chunks)
    y_ = np.array(y_.get_array_of_samples())
    y_ = malaya_speech.astype.int_to_float(y_)
    dir_path = dir_path + fname
    sf.write(dir_path, data=y_, samplerate=sr)
    zero_pad(dir_path)


def pre_process_all():
    train_df = pd.read_csv("../data/test_train.csv")
    dir_path = "../test_data/train/"
    for num, fname in enumerate(train_df["path"]):
        path = "../data/train/" + fname
        try:
            pre_process(path, fname, dir_path)
        except:
            pad_ms = 5000  # Add here the fix length you want (in milliseconds)
            silence = AudioSegment.silent(duration=pad_ms)
            silence.export(dir_path+fname, format='wav')

    # ==============================================================================
    private_df = pd.read_csv("../data/test_private.csv")
    dir_path = "../test_data/private/"
    for num, fname in enumerate(private_df["path"]):
        path = "../data/private/" + fname
        try:
            pre_process(path, fname, dir_path)
        except:
            pad_ms = 5000  # Add here the fix length you want (in milliseconds)
            silence = AudioSegment.silent(duration=pad_ms)
            silence.export(dir_path+fname, format='wav')

    # ==============================================================================
    public_df = pd.read_csv("../data/test_public.csv")
    dir_path = "../test_data/public/"
    for num, fname in enumerate(public_df["path"]):
        path = "../data/public/" + fname
        try:
            pre_process(path, fname, dir_path)
        except:
            pad_ms = 5000  # Add here the fix length you want (in milliseconds)
            silence = AudioSegment.silent(duration=pad_ms)
            silence.export(dir_path+fname, format='wav')


if __name__ == "__main__":
    pre_process_all()
