import os

import numpy as numpy
import pandas as pandas
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

###############################
#   Import Templated Classes  #
###############################


import Modelcomponents
import Lossfunctions


def loss_func(target, pred):
    mse = tf.mse_loss(target, pred)
    esr = Lossfunctions.esr_loss(target, pred)
