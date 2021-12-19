import os
import pandas as pd
import numpy as np
from keras.models import load_model


def prepare_test():
    mfcc = pd.read_csv("../test_data/private_features/mfcc_features.csv")
    test_df = mfcc.iloc[:, :]
    X_test = test_df.iloc[:, :].values.reshape(test_df.shape[0], 13, -1)
    print(test_df.shape)
    X_test = X_test[..., np.newaxis]
    print(X_test.shape)
    return X_test


def predict(re_model, results):
    test_df = pd.read_csv("../data/private_metadata.csv")
    X_test = prepare_test()
    preds = re_model.predict(X_test)
    submission = pd.DataFrame()
    submission["uuid"] = test_df["uuid"]
    submission["assessment_result"] = preds
    submission.to_csv("results.csv", index=0)


def predict_mean():
    test_df = pd.read_csv("../data/private_metadata.csv")
    X_test = prepare_test()
    res = np.zeros(X_test.shape[0])
    res = res[..., np.newaxis]
    print(res.shape)
    model_list = os.listdir("../weights/models/")
    for name in model_list:
        model = load_model("../weights/models/" + name)
        res += model.predict(X_test)
    res /= len(model_list)
    submission = pd.DataFrame()
    submission["uuid"] = test_df["uuid"]
    submission["assessment_result"] = res
    return submission


if __name__ == "__main__":
    submission = predict_mean()
    submission.to_csv("results.csv", index=0)
