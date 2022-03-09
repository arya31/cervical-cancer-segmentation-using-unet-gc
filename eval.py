
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_coef, iou
from train import load_dset,GCRMSprop

H = 256
W = 256


def image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x_bck = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)   ## (1, 256, 256, 3)
    return x_bck, x

def mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x_bck = x
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.int32)
    return x_bck, x

def result(x_bck, y_bck, y_prediction, save_dir):
    line = np.ones((H, 10, 3)) * 255

    y_bck = np.expand_dims(y_bck, axis=-1) ## (256, 256, 1)
    y_bck = np.concatenate([y_bck, y_bck, y_bck], axis=-1) ## (256, 256, 3)

    y_prediction = np.expand_dims(y_prediction, axis=-1)
    y_prediction = np.concatenate([y_prediction, y_prediction, y_prediction], axis=-1) * 255.0

    concat_image = np.concatenate([x_bck, line, y_bck, line, y_prediction], axis=1)
    cv2.imwrite(save_dir, concat_image)

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    """ Load Model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef,'GCRMSprop':GCRMSprop(1e-4)}):
    #with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Dataset """
    dset_location = r"C:\Users\Aryashree mazumder\Desktop\unet\Cell-Nuclei-Segmentation-in-TensorFlow-2.0-main\UNET\img"
    (t_x, t_y), (v_x, v_y), (ts_x, ts_y) = load_dset(dset_location)

    """ Prediction and metrics values """
    SCORE = []
    for x, y in tqdm(zip(ts_x, ts_y), total=len(ts_x)):
        name = x.split("\\")[-1]

        """ Reading the image and mask """
        x_bck, x = image(x)
        y_bck, y = mask(y)

        """ Prediction """
        y_prediction = model.predict(x)[0] > 0.5
        y_prediction = np.squeeze(y_prediction, axis=-1)
        y_prediction = y_prediction.astype(np.int32)

        save_dir = f"results/{name}"
        result(x_bck, y_bck, y_prediction, save_dir)

        """ Flattening the numpy arrays. """
        y = y.flatten()
        y_prediction = y_prediction.flatten()

        """ Calculating metrics values """
        acc = accuracy_score(y, y_prediction)
        f1 = f1_score(y, y_prediction, labels=[0, 1], average="binary")
        jac = jaccard_score(y, y_prediction, labels=[0, 1], average="binary")
        recall = recall_score(y, y_prediction, labels=[0, 1], average="binary")
        precision = precision_score(y, y_prediction, labels=[0, 1], average="binary")
        SCORE.append([name, acc, f1, jac, recall, precision])

    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    """ Saving all the results """
    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")
