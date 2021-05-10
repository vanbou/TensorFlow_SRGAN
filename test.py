import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess = tf.s(config=tf.ConfigProto(log_device_placement=True))