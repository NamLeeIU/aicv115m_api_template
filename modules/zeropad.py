from pydub import AudioSegment
import os


def zero_pad(path):
    pad_ms = 5000  # Add here the fix length you want (in milliseconds)
    audio = AudioSegment.from_wav(path)
    if pad_ms > len(audio):
        silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)
        padded = audio + silence  # Adding silence after the audio
        os.remove(path)
        padded.export(path, format='wav')


def train_zpad():
    train_dir = os.listdir("../test_data/train")
    for fname in train_dir:
        path = "../test_data/train/" + fname
        zero_pad(path)


def private_zpad():
    private_dir = os.listdir("../test_data/private")
    for fname in private_dir:
        path = "../test_data/private/" + fname
        zero_pad(path)


def public_zpad():
    public_dir = os.listdir("../test_data/public")
    for fname in public_dir:
        path = "../test_data/public/" + fname
        zero_pad(path)


if __name__ == "__main__":
    train_zpad()
    """extra_zpad()
    public_zpad()
    private_zpad()"""
