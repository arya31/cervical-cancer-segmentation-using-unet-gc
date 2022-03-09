import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import Recall, Precision
from model import unet
from metrics import dice_coef, iou

H = 256
W = 256

class GCRMSprop(RMSprop):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

def image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def mask(path):
    path = path.decode()
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (W, H))
    y = y/255.0
    y = y.astype(np.float32)
    y = np.expand_dims(y, axis=-1)
    return y

def tensor_parse(x, y):
    def _parse(x, y):
        x = image(x)
        y = mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tensor_dset(X, Y, batch=8):
    dset = tf.data.Dataset.from_tensor_slices((X, Y))
    dset = dset.map(tensor_parse)
    dset = dset.batch(batch)
    dset = dset.prefetch(10)
    return dset

def load_dset(path, split=0.2):
    image_extension = "*.jpg"
    images = sorted(glob(os.path.join(path, "images", image_extension)))
    masks = sorted(glob(os.path.join(path, "masks", image_extension)))
    size = int(len(images) * split)
    print(len(images),len(masks))
    t_x, v_x = train_test_split(images, test_size=size, random_state=42)
    t_y, v_y = train_test_split(masks, test_size=size, random_state=42)

    t_x, ts_x = train_test_split(t_x, test_size=size, random_state=42)
    t_y, ts_y = train_test_split(t_y, test_size=size, random_state=42)

    return (t_x, t_y), (v_x, v_y), (ts_x, ts_y)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory to save files """
    if not os.path.exists("files"):
        os.makedirs("files")

    """ Hyperparaqmeters """
    batches = 8
    lr = 1e-4   ## 0.0001
    n_epochs = 10
    model_location = "files/model.h5"
    csv_location = "files/data.csv"

    """ Dataset """
    dset_location = r".\img\DSB"

    (t_x, t_y), (v_x, v_y), (ts_x, ts_y) = load_dset(dset_location)
    t_x, t_y = shuffle(t_x, t_y)

    print(f"Train: {len(t_x)} - {len(t_y)}")
    print(f"Valid: {len(v_x)} - {len(v_y)}")
    print(f"Test: {len(ts_x)} - {len(ts_y)}")

    t_dset = tensor_dset(t_x, t_y, batch=batches)
    v_set = tensor_dset(v_x, v_y, batch=batches)

    t_steps = len(t_x)//batches
    valid_steps = len(v_x)//batches

    if len(t_x) % batches != 0:
        t_steps += 1

    if len(v_x) % batches != 0:
        valid_steps += 1

    """ Model """
    model = unet((H, W, 3))
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=GCRMSprop(lr), metrics=metrics)
    #model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
    callbacks = [
        ModelCheckpoint(model_location, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_location),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        t_dset,
        epochs=n_epochs,
        validation_data=v_set,
        steps_per_epoch=t_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )
