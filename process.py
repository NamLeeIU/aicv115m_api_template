import os
from pydub import AudioSegment

from pydub import AudioSegment
import os
import librosa
import pandas as pd
import numpy as np
import malaya_speech
from pydub.silence import split_on_silence
import soundfile as sf
from keras.models import load_model
import os
from pydub import AudioSegment
from configs.config import Config


def convert_to_wav(file_path):
    """
    This function is to convert an audio file to .wav file

    Args:
        file_path (str): paths of audio file needed to be convert to .wav file

    Returns:
        new path of .wav file
    """
    ext = file_path.split(".")[-1]
    assert ext in [
        "mp4", "mp3", "acc"], "The current API does not support handling {} files".format(ext)

    sound = AudioSegment.from_file(file_path, ext)
    wav_file_path = ".".join(file_path.split(".")[:-1]) + ".wav"
    sound.export(wav_file_path, format="wav")

    os.remove(file_path)
    return wav_file_path


def pre_process(path):
    y, sr = librosa.load(path, res_type='kaiser_fast')
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
    os.remove(path)
    sf.write(path, data=y_, samplerate=sr)


def pre_process_all(df):
    path = df["path"][0]
    try:
        pre_process(path)

    except:
        pad_ms = 5000  # Add here the fix length you want (in milliseconds)
        silence = AudioSegment.silent(duration=pad_ms)
        silence.export(path, format='wav')


def zero_pad(path):
    pad_ms = 5000  # Add here the fix length you want (in milliseconds)
    audio = AudioSegment.from_wav(path)
    if pad_ms > len(audio):
        silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)
        padded = audio + silence  # Adding silence after the audio
        os.remove(path)
        padded.export(path, format='wav')


def zpad_all(df):
    for path in df["path"]:
        zero_pad(path)


def feature(path):
    source, sr = librosa.load(path, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=source[0:sr*5], sr=sr, n_mfcc=13)
    return mfcc


def extract_all(df):
    Xmfcc = []
    for path in df["path"]:
        mfcc = feature(path)
        mfcc = mfcc.reshape(-1,)
        Xmfcc.append(mfcc)
    return Xmfcc


def extract_data(df):
    Xmfcc = extract_all(df)
    mfcc_df = pd.DataFrame(Xmfcc)
    return mfcc_df


def prepare_test(df):
    mfcc = df
    test_df = mfcc.iloc[:, :]
    X_test = test_df.iloc[:, :].values.reshape(test_df.shape[0], 13, -1)
    print(test_df.shape)
    X_test = X_test[..., np.newaxis]
    print(X_test.shape)
    return X_test


def predict_mean(df):
    X_test = prepare_test(df)
    res = np.zeros(X_test.shape[0])
    res = res[..., np.newaxis]
    model_list = os.listdir("../weights")
    for name in model_list:
        model = load_model("../weights/" + name)
        preds = model.predict(X_test)
        res += model.predict(X_test)
    res /= len(model_list)
    positive_proba = res
    return positive_proba


def predict(df):
    meta_df = df
    df = pd.DataFrame()
    df["path"] = meta_df["file_path"]
    pre_process_all(df)
    zpad_all(df)
    mfcc_df = extract_data(df)
    positive_proba = predict_mean(mfcc_df)
    return positive_proba[0]
