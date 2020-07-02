#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   keras_segmentation.py

   train NN model for segmentation using keras and tensorflow

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import platform
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore",category=DeprecationWarning)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

useHorovod = False

import keras
import zarr
import argparse
import numpy as np
import tensorflow as tf

if useHorovod:
  import horovod.keras as hvd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.utils import Sequence
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TensorBoard

from utils.utils import usage
from learning.models.unet import unet
from learning.losses.losses import dice_coeff, bce_dice_loss, jaccard_coef, jaccard_loss, dice_loss, tversky_coeff, tversky_loss, focal_loss
from learning.learning_utils import get_model_memory_usage

headless = True
#24 for bands up to 3
#batch_size = 20 
batch_size = 14
num_classes = 2
epochs = 1000

global model_name
model_name = "unet_ci"

import logging

logging_format = '%(asctime)s - %(name)s - %(message)s'
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, 
    format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("KerasSegmentation")


# Horovod: initialize Horovod.
if useHorovod:
  hvd.init()
  logger.info("horovod size: %s  rank: %s  device rank: %s  host: %s", hvd.size(), hvd.rank(), hvd.local_rank(), platform.node())

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

if useHorovod and gpus:
  tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
 
if len(tf.config.list_physical_devices('GPU')) == 0:
  logger.error("ERROR: No cuda detected")
  logger.error("ERROR:  %s", platform.node())
  #exit(-1)

class generator(Sequence):
    def __init__(self, x_set, y_set, batch_size, limit=None, bands=[0,1,2,3,4,5,6]):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.limit = limit
        self.bands = bands

        if self.limit is None:
          self.length = self.x.shape[0]
        else:
          self.length = min(self.limit, self.x.shape[0])

        self.order = np.random.permutation(self.x.shape[0])[0:self.length]

    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))

    def __getitem__(self, idx):
      
        cutStart = idx * self.batch_size
        cutEnd = min((idx + 1) * self.batch_size, self.limit)

        indices = self.order[cutStart:cutEnd]

        # B08,B09,B10,B11,B13,B14
        # 0, B08 Upper-level water vapor
        # 1, B09 Mid-level water vapor
        # 2, B10 Low-level water vapor
        # 3, B11 Cloud Top Phase
        # 4, B13 clean IR
        # 5, B14 longwave IR
        # 6, B16 longwave IR 13.3
   
        train = np.array(self.x.oindex[indices])
        test = np.array(self.y.oindex[indices])
   
        timeperiods = 3
        total_bands = 7
        bands_wanted = self.bands 
        band_indices = []
        for i in range(0,timeperiods):
           for b in range(0, len(bands_wanted)):
                band_indices.append(i*total_bands+bands_wanted[b])
   
        train = train[:, :, :, band_indices]
        # reverse array, oldest at the begginning
        train = train[:, :, :, ::-1]
        test = test[:, :, :, :]

        train = (train-128.0)/255.0
        test[test<=35.0] = 0
        test[test>35] = 1
        test = test.astype(np.uint8)

        return train, test 

def readData(file):
  if not useHorovod or hvd.rank() == 0:
     logger.info("reading in training %s", file)
  loaded_data = zarr.open(file, mode='r')
  x_train = loaded_data['train']
  y_train = loaded_data['test']

  return x_train,y_train

def main(lossfunction="tversky", lossrate=1e-4, depth=7, optimizer="rms", n_filters=32, fixed=False, resnet=False, bands=[0,1,2,3,4,5], batchnorm=True, dropout=False, dropout_rate=0.10, noise=False, noise_rate=0.1, ramp=False, earlystop=False):

  verbose = 0
  if not useHorovod:
    verbose = 1
  elif hvd.rank() == 0:
    verbose = 2
    
  if verbose > 0:
    logger.info("using bands %s", bands)

  train_file = '../../data/train'
  val_file = '../../data/val'
  test_file = '../../data/test'

  #basepath = '/scratch2/BMC/gsd-hpcs/Jebb.Q.Stewart/git/gsd-machine-learning/src/cwb/ci/data/'
  #train_file = basepath + '/cwbci_512_l30_7bands_radar_train'
  #val_file = basepath + '/cwbci_512_l30_7bands_radar_val'
  #test_file = basepath + '/cwbci_512_l30_7bands_radar_test'

  if verbose > 0:
     logger.info('reading in train data')

  x_train, y_train = readData(train_file)
  x_val, y_val = readData(val_file)
  x_test, y_test = readData(test_file)

  #sample = np.min(100, x_train.shape[0])
  sample = 100
  if verbose > 0:
     logger.info("Sample data: ")
     logger.info("  Training input : max[0]: %s", np.max(x_train[sample,:,:,0]))
     logger.info("                   min[0]: %s", np.min(x_train[sample,:,:,0]))
     logger.info("           shape : %s", x_train.shape)
     logger.info("  Train labels: max[0]: %s min[1]: %s  dtype: %s", np.max(y_train[sample,:,:,:]), np.min(y_train[sample,:,:,:]), y_train.dtype)
     logger.info("  using loss function %s", lossfunction)


  loss = tversky_loss(alpha=0.3, beta=0.7)
  if lossfunction == "dice":
     loss = dice_loss
  elif lossfunction == "tversky2":
     loss = tversky_loss2(alpha=0.7, beta=0.3)
  elif lossfunction == "tversky3":
     loss = tversky_loss2(alpha=0.2, beta=0.8)
  elif lossfunction == "bcedice":
     loss = bce_dice_loss
  elif lossfunction == "focal":
     loss = focal_loss3(gamma=2)
  elif lossfunction == "bce":
     loss = 'binary_crossentropy'
  elif lossfunction == "mse":
     loss = 'mse'
  elif lossfunction == "rmse":
     loss = 'rmse'


  channels = len(bands)*3
  model = unet(img_rows=512, img_cols=512, channels=channels, output_channels=1, fixed=fixed, 
               batchnorm=batchnorm, resnet=resnet, n_filters=n_filters, depth=depth, 
               dropout=dropout, dropout_rate=dropout_rate, noise=noise, 
               noise_rate=noise_rate, final_activation='sigmoid', verbose=verbose)

  if useHorovod:
    opt = hvd.DistributedOptimizer(RMSprop(lr=lossrate*hvd.size()))
    if optimizer == "adam":
      opt = hvd.DistributedOptimizer(Adam(lr=lossrate*hvd.size()))
  else:
    opt = RMSprop(lr=lossrate)
    if optimizer == "adam":
      opt = Adam(lr=lossrate)

  model.compile(optimizer=opt, loss=loss, metrics=[tversky_coeff(alpha=0.3, beta=0.7), dice_coeff, 'accuracy'])

  if verbose > 0:
 
     logger.info("Model Summary:\n%s", model.summary())
     logger.info("Estimated Model GPU usage: %s GB", get_model_memory_usage(batch_size, model))
     logger.info("Current host memory usage: %s", usage());

     # serialize model to JSON
     model_json = model.to_json()
     if not os.path.isdir("models"):
         os.makedirs("models")

     model_file = "models/" + model_name + ".json"
     with open(model_file, "w") as json_file:
        json_file.write(model_json)
     logger.info("saved model to %s", model_file)

  callbacks = []
  if useHorovod:
     callbacks.append( hvd.callbacks.BroadcastGlobalVariablesCallback(0))
     callbacks.append( hvd.callbacks.MetricAverageCallback())
     callbacks.append( hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=verbose))

  if earlystop:
              callbacks.append(EarlyStopping(monitor='val_loss',
                         patience=30,
                         verbose=verbose,
                         min_delta=1e-4,
                         restore_best_weights=True))

  if ramp:
        if useHorovod:
              # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
              callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=15, end_epoch=40, multiplier=1.))
              callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=40, end_epoch=70, multiplier=1e-1))
              callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=70, end_epoch=100, multiplier=1e-2))
              callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=100, multiplier=1e-3))

              # Reduce the learning rate if training plateaues.
              #keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)]
              #ReduceLROnPlateau(monitor='val_loss',
              #               factor=0.1,
              #               patience=4,
              #               verbose=1,
              #               min_delta=1e-4),

  training_bg = generator(x_train, y_train, batch_size, limit=80000, bands=bands)
  val_bg = generator(x_val, y_val, batch_size, limit=40000, bands=bands)
  test_bg = generator(x_test, y_test, batch_size, limit=20000, bands=bands)

  if useHorovod: 
    training_bg.order = hvd.broadcast(training_bg.order, 0, name='training_bg_order').numpy()
    val_bg.order = hvd.broadcast(val_bg.order, 0, name='val_bg_order').numpy()
    test_bg.order = hvd.broadcast(test_bg.order, 0, name='test_bg_order').numpy()

  if verbose > 0:
     logger.info("Training size: %s : steps : %s", training_bg.length, (training_bg.length//batch_size))
     logger.info("Validation size: %s : steps : %s", val_bg.length, (val_bg.length//batch_size))

  # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
  if not useHorovod or hvd.rank() == 0:
    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoints/' + model_name + '_checkpoint-{epoch:02d}-{val_loss:.3f}.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True))
    #callbacks.append(keras.callbacks.TensorBoard(log_dir='tflogs'))

  size = 1
  if useHorovod:
    size = hvd.size()

  history = model.fit_generator(generator=training_bg, 
         steps_per_epoch=(training_bg.length//batch_size) // size,
         epochs=epochs,
         verbose=verbose,
         callbacks=callbacks,
         validation_data=val_bg,
         validation_steps=(val_bg.length // batch_size) // size,
         shuffle=True,
         use_multiprocessing=False,
         workers=2,
         max_queue_size=8)
  
  if not useHorovod or hvd.rank() == 0:
     # serialize weights to HDF5
     logger.info("saving weights")
     if not os.path.isdir("weights"):
        os.makedirs("weights")

     weights_file = "weights/" + model_name + ".h5"
     model.save_weights(weights_file)
     logger.info("Saved weights to disk %s", weights_file)

     logger.info("evaluating results")

  scores = model.evaluate_generator(generator=test_bg, steps=(test_bg.length//batch_size) // size, workers=2,
         max_queue_size=8, use_multiprocessing=False, verbose=verbose)

  if not useHorovod or hvd.rank() == 0:

    logger.info('Test scores: %s', scores)
    if not os.path.isdir("images"):
        os.makedirs("images")
  
    # plt.xkcd()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("images/" + model_name +"_acc.png")
  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("images/" + model_name +"_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Keras Segmentation")

    parser.add_argument('-n', '--name',
       dest="name", required=True,
       help="name of output prefix for saving model, weights, images, etc...")

    parser.add_argument('-l', '--loss',
       dest="loss", required=False, default="tversky",
       help="loss function to use")

    parser.add_argument('-lr', '--lossrate',
       dest="lossrate", required=False, default=1e-4, type=float,
       help="loss rate")

  
    parser.add_argument('-o', '--optimizer',
       dest="optimizer", required=False, default="rms",
       help="optimizer to use")

    parser.add_argument('-b', '--bands', 
       dest='bands', required=False, default=[0,1,2,3,4,5], type=int, nargs='+',
       help="bands to use 0-5 (ie 6 total bands)")

    parser.add_argument('-bc', '--batch',
       dest="batch_size", required=False, default=8, type=int,
       help="batch size")

    parser.add_argument('-bn', '--batchnorm',
       dest="batchnorm", required=False, action='store_true',
       help="use batchnorm")

    parser.add_argument('-nf', '--n_filters', 
       dest='n_filters', required=False, default=64, type=int, 
       help="number of filters to use")
 
    parser.add_argument('-d', '--depth', 
       dest='depth', required=False, default=7, type=int,
       help="depth of NN")
 
    parser.add_argument('-rn', '--resnet',
       dest="resnet", required=False, action='store_true',
       help="use resnet")
 
    parser.add_argument('-f', '--fixed',
       dest="fixed", required=False, action='store_true',
       help="use fixed NN") 

    parser.add_argument('-e', '--epochs',
       dest='epochs', required=False, default=100, type=int,
       help="number of epochs")

    parser.add_argument('--dropout', dest='dropout', required=False, action='store_true', help='use dropout')

    parser.add_argument('--dropout_rate',
       dest="dropout_rate", required=False, default=0.1, type=float,
       help="dropout rate")

    parser.add_argument('--noise', dest='noise', required=False, action='store_true', help='use noise')

    parser.add_argument('--noise_rate',
       dest="noise_rate", required=False, default=0.1, type=float,
       help="noise rate")

    parser.add_argument('--ramp', dest='ramp', required=False, action='store_true', help='use ramp')
    parser.add_argument('--earlystop', dest='earlystop', required=False, action='store_true', help='use earlystop')

    args = parser.parse_args()

    model_name = args.name
    loss = args.loss.lower()
    lossrate = args.lossrate
    optimizer = args.optimizer.lower()
    bands = args.bands
    n_filters = args.n_filters
    depth = args.depth
    resnet = args.resnet
    fixed = args.fixed
    batchnorm = args.batchnorm
    epochs = args.epochs
    batch_size = args.batch_size
    dropout = args.dropout
    dropout_rate = args.dropout_rate
    noise = args.noise
    noise_rate = args.noise_rate
    earlystop = args.earlystop
    ramp = args.ramp

    if not useHorovod or hvd.rank() == 0:
       logger.info("running with loss function: %s and loss rate of %s using optimizer %s", loss, lossrate, optimizer)
       logger.info("  saving info with prefix of %s", model_name)
       logger.info("  batch size: %s", batch_size)

    main(lossfunction=loss, lossrate=lossrate, optimizer=optimizer, n_filters=n_filters, batchnorm=batchnorm, resnet=resnet,
         fixed=fixed, depth=depth, dropout=dropout, dropout_rate=dropout_rate, noise=noise, noise_rate=noise_rate, earlystop=earlystop, ramp=ramp, bands=bands)

