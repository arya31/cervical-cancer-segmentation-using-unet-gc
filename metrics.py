
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def iou(y_true, y_prediction):
    def f(y_true, y_prediction):
        intersection = (y_true * y_prediction).sum()
        union = y_true.sum() + y_prediction.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_prediction], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_prediction):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_prediction = tf.keras.layers.Flatten()(y_prediction)
    intersection = tf.reduce_sum(y_true * y_prediction)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_prediction) + smooth)

def dice_loss(y_true, y_prediction):
    return 1.0 - dice_coef(y_true, y_prediction)
