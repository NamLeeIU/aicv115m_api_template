import os
import pandas as pd


def build_test_path():
    train_meta_df = pd.read_csv("../data/train_metadata.csv")
    train_meta_df = train_meta_df.loc[train_meta_df["audio_noise_note"].isnull(
    )]

    train_df = pd.DataFrame()
    train_df["path"] = train_meta_df["uuid"].apply(
        lambda uuid: f"../test_data/train/{uuid}.wav")
    train_df["label"] = train_meta_df["assessment_result"]

    # ==============================================================================

    extra_dir = os.listdir("../data/extra")
    extra_df = pd.DataFrame()
    extra_df["path"] = ["../test_data/train/" + fname for fname in extra_dir]
    extra_df["label"] = 1

    # ==============================================================================

    train_df = pd.concat([train_df, extra_df], axis=0)
    train_df.to_csv("../test_data/test_train.csv", index=False)

    # ==============================================================================

    private_df = pd.read_csv("../data/private_metadata.csv")
    private_df["assessment_result"] = private_df["uuid"].apply(
        lambda uuid: f"../test_data/private/{uuid}.wav")
    private_df.columns = ["uuid", "path"]
    private_df.to_csv("../test_data/test_private.csv", index=False)

    # ==============================================================================
    public_df = pd.read_csv("../data/public_metadata.csv")
    public_df["assessment_result"] = public_df["uuid"].apply(
        lambda uuid: f"../test_data/public/{uuid}.wav")
    public_df.columns = ["uuid", "path"]
    private_df.to_csv("../test_data/test_public.csv", index=False)


def build_preprocess_path():
    train_meta_df = pd.read_csv("../data/train_metadata.csv")
    train_meta_df = train_meta_df.loc[train_meta_df["audio_noise_note"].isnull(
    )]

    train_df = pd.DataFrame()
    train_df["path"] = train_meta_df["uuid"].apply(lambda uuid: f"{uuid}.wav")

    # ==============================================================================

    extra_dir = os.listdir("../data/extra")
    extra_df = pd.DataFrame()
    extra_df["path"] = [fname for fname in extra_dir]

    # ==============================================================================

    train_df = pd.concat([train_df, extra_df], axis=0)
    train_df.to_csv("../data/test_train.csv", index=False)

    # ==============================================================================

    private_meta_df = pd.read_csv("../data/private_metadata.csv")
    private_df = pd.DataFrame()
    private_df["path"] = private_meta_df["uuid"].apply(
        lambda uuid: f"{uuid}.wav")
    private_df.to_csv("../data/test_private.csv", index=False)

    # ==============================================================================
    public_meta_df = pd.read_csv("../data/public_metadata.csv")
    public_df = pd.DataFrame()
    public_df["path"] = public_meta_df["uuid"].apply(
        lambda uuid: f"{uuid}.wav")
    public_df.to_csv("../data/test_public.csv", index=False)


if __name__ == "__main__":
    build_test_path()
    """build_preprocess_path()"""
