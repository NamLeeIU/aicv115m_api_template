import librosa
import pandas as pd


def mean_feature(path):
    source, sr = librosa.load(path, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=source[0:5*sr], sr=sr, n_mfcc=13)
    return mfcc


def extract_all(df):
    Xmfcc = []
    for path in df["path"]:
        mfcc = mean_feature(path)
        mfcc = mfcc.reshape(-1,)
        Xmfcc.append(mfcc)
    return Xmfcc


def extract_train():
    train_df = pd.read_csv("../test_data/test_train.csv")
    Xmfcc = extract_all(train_df)
    mfcc_df = pd.DataFrame(Xmfcc)
    mfcc_df["label"] = train_df["label"]
    mfcc_df.to_csv(
        "../test_data/train_features/mfcc_features.csv", index=False)


def extract_private():
    private_df = pd.read_csv("../test_data/test_private.csv")
    Xmfcc_private = extract_all(private_df)
    mfcc_df_private = pd.DataFrame(Xmfcc_private)
    mfcc_df_private.to_csv(
        "../test_data/private_features/mfcc_features.csv", index=False)


def extract_public():
    public_df = pd.read_csv("../test_data/test_public.csv")
    Xmfcc_public = extract_all(public_df)
    mfcc_df_public = pd.DataFrame(Xmfcc_public)
    mfcc_df_public.to_csv(
        "../test_data/public_features/mfcc_features.csv", index=False)


if __name__ == "__main__":
    extract_train()
    extract_private()
    extract_public()
