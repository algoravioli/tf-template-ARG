import tensorflow as tf
import numpy as np
import librosa


def pre_emphasis_filter(x, coeff=0.85):
    return tf.concat([x[0:1], x[1:] - coeff * x[:-1]], axis=0)


eps = np.finfo(float).eps


def esr_loss(target_y, pred_y, emphasis_func=lambda x: x):
    target_yp = emphasis_func(target_y)
    pred_yp = emphasis_func(pred_y)
    mse = tf.math.reduce_sum(tf.math.square(target_yp - pred_yp))
    energy = tf.math.reduce_sum(tf.math.square(target_yp))

    loss_unnorm = mse / tf.cast(energy + eps, tf.float32)
    N = tf.cast((tf.shape(target_y)[0] * tf.shape(target_y)[1]), tf.float32)
    return tf.sqrt(loss_unnorm / N)


def esr_loss_with_emph(target, pred):
    esr_with_emph_loss = esr_loss(target, pred, pre_emphasis_filter)
    return esr_with_emph_loss


def fft_loss(target_y, pred_y):
    target_fft = np.sum(np.abs(librosa.stft(target_y)))
    predict_fft = np.sum(np.abs(librosa.stft(pred_y)))
    loss = np.abs(predict_fft - target_fft)
    return loss


def mse_loss(target_y, pred_y):
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss_amount = mse(target_y, pred_y).numpy()
    return mse_loss_amount


def avg_loss(target_y, pred_y):
    target_mean = tf.math.reduce_mean(target_y)
    pred_mean = tf.math.reduce_mean(pred_y)
    return tf.math.abs(target_mean - pred_mean)


def bounds_loss(target_y, pred_y):
    target_min = tf.math.reduce_min(target_y)
    target_max = tf.math.reduce_max(target_y)
    pred_min = tf.math.reduce_min(pred_y)
    pred_max = tf.math.reduce_max(pred_y)
    return tf.math.abs(target_min - pred_min) + tf.math.abs(target_max - pred_max)
