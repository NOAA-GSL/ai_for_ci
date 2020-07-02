#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   test_model.py

   Test ML Model for t-x lead time for and generate images of inputs, and predicted outputs

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

"""

from keras.models import model_from_json

import keras
import argparse
import os
import sys

from PIL import Image
import numpy as np

from process_data import processTimeBlock
from process_data import processTime as processCurrent
import dateutil.parser
from datetime import timedelta
import matplotlib
# Needed for headless matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import logging

logging_format = '%(asctime)s - %(name)s - %(message)s'
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, 
    format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("TestModel")

# setup labels colormap with transparency for overlay and 
jet = cm.jet
my_cmap = jet(np.arange(jet.N))
my_cmap[:,-1] = 0.5
my_cmap[0,-1] = 0.15

# Create new colormap
jet = matplotlib.colors.ListedColormap(my_cmap)
jet.set_under([0,0,0,0])

# dependent on input data
total_bands = 7

model_name = ""

def getDataForBands(train, timeperiods=3, total_bands=7, bands=[0,1,2,3,4,5,6]):
    bands_wanted = bands
    band_indices = []

    # construct array of indices into data array for bands 
    for i in range(0,timeperiods):
        for b in range(0, len(bands_wanted)):
            band_indices.append(i*total_bands+bands_wanted[b])

    train = train[:, :, :, band_indices]
    return train

def inference(model, validTime, bands, outdir="test", lead_time=90):

    print ('working on ', validTime)
    try:
      # get block used for inference with a lead_time
      logger.info("Getting block for inference")
      train, testp, time = processTimeBlock(validTime, lead_time=lead_time, width=512, height=512)
     
      # get observations at T
      logger.info("Getting observations at T")
      obs_sat, obs_rad, time = processCurrent(validTime, width=512, height=512)

      if train is not None and obs_rad is not None:
         logger.info("Reducing block to requested bands")
         train = getDataForBands(train, timeperiods=3, bands=bands)
         logger.info("   resulting data shape: %s", train.shape)

         logger.info("Reducing obs to requested bands")
         obs_sat = getDataForBands(obs_sat, timeperiods=1, bands=bands)
         logger.info("   obs shape :: %s", obs_sat.shape)

         labelfile = validTime.strftime(outdir + "/labels-%Y-%m-%dT%H:%M:00.png")
         truthfile = validTime.strftime(outdir + "/truth-%Y-%m-%dT%H:%M:00.png")
         predictfile = validTime.strftime(outdir + "/predict-%Y-%m-%dT%H:%M:00.png")
         combofile = validTime.strftime(outdir + "/combo-%Y-%m-%dT%H:%M:00.png")
 
         count = 0
         for b in bands:
            logger.info("creating image for sat band %s", b)
            sat = obs_sat[0,:,:,count]
            logger.info("min max for band %s: %s  ::  %s", b, np.max(sat), np.min(sat))
            satlabel = "sat_band_" + str(b)
            satfile = validTime.strftime(outdir + "/" + satlabel + "-%Y-%m-%dT%H:%M:00.png")
            sat_img = Image.fromarray(np.uint8(sat))
            sat_img.save(satfile)
            count += 1
      
         train = (train-128.0)/255.0
         logger.debug("%s :: %s :: %s :: %s", validTime, np.min(obs_rad), np.max(obs_rad), np.mean(obs_rad))
         obs_rad[obs_rad<=35.0] = 0
         logger.debug("%s :: %s :: %s :: %s", validTime, np.min(obs_rad), np.max(obs_rad), np.mean(obs_rad))
         obs_rad[obs_rad>35] = 1
         obs_rad = obs_rad.astype(np.uint8)
         logger.debug("%s :: %s :: %s :: %s", validTime, np.min(obs_rad), np.max(obs_rad), np.mean(obs_rad))

         # generate labels image
         labels = obs_rad[0,:,:,0]
         tmpdata = (labels*255).astype(np.uint8)
         lab_img = Image.fromarray(jet(labels.astype(np.float32), bytes=True))
         lab_img.save(labelfile)
     
         # run model
         predict = model.predict(train)

         # generate predicted labels image
         pred_img = Image.fromarray(jet(predict[0,:,:,0], bytes=True))
         pred_img.save(predictfile)

         # combine labels and observation into truth image
         sat_img = sat_img.convert('RGBA')
         truth_img = Image.alpha_composite(sat_img, lab_img)
         truth_img.save(truthfile)

         # combine predicted labels and observation into predict_image
         predict_img = Image.alpha_composite(sat_img, pred_img)

         # combine side by side truth and prediction
         imgs_comb = np.hstack( (np.asarray(i) for i in [truth_img, predict_img]))
         imgs_comb = Image.fromarray( imgs_comb)
         imgs_comb.save(combofile)

    except:
      logger.error(sys.exc_info())


def main():

    parser = argparse.ArgumentParser("Test Model")

    parser.add_argument('-n', '--name',
       dest="name", required=True,
       help="name of model without suffix")

    parser.add_argument('-mp', '--modelpath',
       dest="modelpath", required=False, default='models',
       help="path to find model json file")

    parser.add_argument('-wp', '--weightpath',
       dest="weightpath", required=False, default='weights',
       help="path to find model weights file")

    parser.add_argument('-o', '--outputpath',
       dest="output", required=False, default='test',
       help="path to save files")

    parser.add_argument('-s', '--start',
       dest="start", required=True, default=None,
       help="start iso time")

    parser.add_argument('-e', '--end',
       dest="end", required=True, default=None, type=str,
       help="end iso time")

    parser.add_argument('-b', '--bands', 
       dest='bands', required=False, default=[0,1,2,3,4,5], type=int, nargs='+',
       help="bands to use 0-5 (ie 6 total bands)")

    args = parser.parse_args()

    model_name = args.name
    weight_path = args.weightpath
    model_path = args.modelpath
    outdir = args.output
    start = args.start
    end = args.end
    bands = args.bands
    
    model_file = model_path + "/" + model_name + '.json'
    logger.info("loading model from file: %s", model_file)
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    weights_file = weight_path + "/" + model_name + ".h5"
    logger.info("loading weights from file: %s", weights_file)
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")

    if not os.path.isdir(outdir):
       os.makedirs(outdir)

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

    #timestamp = "2018-08-12 12:00:00"
    #timestamp = "2018-05-20 00:00:00"
    validTime = dateutil.parser.parse(start)

    #endstamp = "2018-08-13 12:10:00"
    #endstamp = "2018-05-20 23:59:00"
    endTime = dateutil.parser.parse(end)

    while validTime < endTime:
        inference(loaded_model, validTime, bands, outdir=outdir)
        validTime = validTime + timedelta(minutes=10)


if __name__ == "__main__":
   main()
