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
from PIL import Image
import numpy as np
import os

from process_data import processTimeBlock
from process_data import processTime as processCurrent
#from process_data import processTime as processCurrent
import dateutil.parser
from datetime import timedelta
import argparse
import matplotlib
# Needed for headless matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
#from mpl_toolkits.basemap import Basemap

import logging


logging_format = '%(asctime)s - %(name)s - %(message)s'
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, 
    format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("TestModel")

jet = cm.jet
my_cmap = jet(np.arange(jet.N))
my_cmap[:,-1] = 0.5
my_cmap[0,-1] = 0.15

# Create new colormap
jet = matplotlib.colors.ListedColormap(my_cmap)
jet.set_under([0,0,0,0])

timeperiods = 3
total_bands = 7

def getDataForBands(train, bands=[0,1,2,3,4,5,6]):
    bands_wanted = bands
    band_indices = []
    for i in range(0,timeperiods):
        for b in range(0, len(bands_wanted)):
            band_indices.append(i*total_bands+bands_wanted[b])

    train = train[:, :, :, band_indices]
    return train

def inference(model, validTime, bands, outdir="test", lead_time=90):

    print ('working on ', validTime)
    try:
      train, testp, time = processTimeBlock(validTime, lead_time=lead_time)
      current_sat, test, time = processCurrent(validTime)

      if train is not None and test is not None:
         train = getDataForBands(train, bands=bands)

         print ("data: ", train.shape)
         labelfile = validTime.strftime(outdir + "/labels-%Y-%m-%dT%H:%M:00.png")
     
         sat1file = validTime.strftime(outdir + "/sat1-%Y-%m-%dT%H:%M:00.png")
         sat2file = validTime.strftime(outdir + "/sat2-%Y-%m-%dT%H:%M:00.png")
         sat3file = validTime.strftime(outdir + "/sat3-%Y-%m-%dT%H:%M:00.png")
         truthfile = validTime.strftime(outdir + "/truth-%Y-%m-%dT%H:%M:00.png")
         predictfile = validTime.strftime(outdir + "/predict-%Y-%m-%dT%H:%M:00.png")
         combofile = validTime.strftime(outdir + "/combo-%Y-%m-%dT%H:%M:00.png")
         combo2file = validTime.strftime(outdir + "/combo2-%Y-%m-%dT%H:%M:00.png")
         combo3file = validTime.strftime(outdir + "/combo3-%Y-%m-%dT%H:%M:00.png")

         #pfile = validTime.strftime(outdir + "pmapped-%Y-%m-%dT%H:%M:00.png")

     
         #tmptrain = train[:,150:-150,150:-150,:]
         tmptest = test[:,150:-150,150:-150,:]
         #tmptestp = testp[:,150:-150,150:-150,:]
         current_sat = current_sat[:, 150:-150, 150:-150, :]

         #train = np.zeros((tmptrain.shape[0],512,512,tmptrain.shape[3]), dtype=tmptrain.dtype)
         test = np.zeros((tmptest.shape[0],512,512,tmptest.shape[3]), dtype=tmptest.dtype)
         real    = np.zeros((current_sat.shape[0],512,512,current_sat.shape[3]), dtype=current_sat.dtype)
         #testp = np.zeros((tmptestp.shape[0],512,512,tmptestp.shape[3]), dtype=tmptestp.dtype)


         for i in range(0, train.shape[0]):
             #for c in range(0, train.shape[3]):
                #train[i,:,:,c] = np.array(Image.fromarray(tmptrain[i,:,:,c]).resize((512, 512), Image.LANCZOS))
             for c in range(0, test.shape[3]):
                test[i,:,:,c] = np.array(Image.fromarray(tmptest[i,:,:,c]).resize((512, 512), Image.LANCZOS))
             #for c in range(0, testp.shape[3]):
                #testp[i,:,:,c] = np.array(Image.fromarray(tmptestp[i,:,:,c]).resize((512, 512), Image.LANCZOS))
             for c in range(0, current_sat.shape[3]):
                real[i,:,:,c] = np.array(Image.fromarray(current_sat[i,:,:,c]).resize((512, 512), Image.LANCZOS))
  #


         print ("shapes:: ", train.shape, " :: ", test.shape)

         sat1 = real[0,:,:,-1]
         print ("min max: ", np.max(sat1), " :: ", np.min(sat1))
         img = Image.fromarray(np.uint8(sat1))
         img.save(sat1file)
         #exit(-9)
         satp = train[0,:,:,-1]
         satp_img = Image.fromarray(np.uint8(satp))
         sat2 = real[0,:,:,-1]
         sat2_img = Image.fromarray(np.uint8(sat2))
         sat2_img.save(sat2file)
         sat3 = real[0,:,:,len(bands)]
         sat3_img = Image.fromarray(np.uint8(sat3))
         sat3_img.save(sat3file)
      
         train = (train-128.0)/255.0
         print (validTime, " :: ", np.min(test), " :: ", np.max(test), " :: ", np.mean(test))
         test[test<=35.0] = 0
         print (validTime, " :: ", np.min(test), " :: ", np.max(test), " :: ", np.mean(test))
         test[test>35] = 1
         test = test.astype(np.uint8)
         print (validTime, " :: ", np.min(test), " :: ", np.max(test), " :: ", np.mean(test))

         labels = test[0,:,:,0]
         print ("labels ", labels.dtype)
         tmpdata = (labels*255).astype(np.uint8)
         print ("tmpdata ", tmpdata.dtype)
         lab_img = Image.fromarray(jet(labels.astype(np.float32), bytes=True))
         #img = toimage(labels.astype(np.float32))
         lab_img.save(labelfile)
     
         predict = model.predict(train)
         #unique, counts = np.unique(predict, return_counts=True)

         #print (np.asarray((unique, counts)).T)
         #print (predict.shape, " :: ", (predict[(predict>0.0) & (predict<0.25)]).sum(), " :: ", (predict[(predict>0.5) & (predict<0.75)]).sum()," :: ", (predict==1).sum())
     
         #print ("data type: ", predict.dtype)
         last_rad_img = Image.fromarray(jet((testp[0,:,:,0]*255).astype(np.uint8), bytes=True))
         pred_img = Image.fromarray(jet(predict[0,:,:,0], bytes=True))
         #pred_img = Image.fromarray(jet((predict[0,:,:,0]*255).astype(np.uint8)))
         pred_img.save(predictfile)

         sat2_img = sat2_img.convert('RGBA')
         truth_img = Image.alpha_composite(sat2_img, lab_img)
         truth_img.save(truthfile)

         predict_img = Image.alpha_composite(sat2_img, pred_img)

         satp_img = satp_img.convert('RGBA')
         guess_img = Image.alpha_composite(satp_img, pred_img)
         cur_img = Image.alpha_composite(satp_img, last_rad_img)

         next_img = Image.alpha_composite(sat2_img, pred_img)
         imgs_comb = np.vstack( (np.asarray(i) for i in [truth_img, next_img]))
         imgs_comb = Image.fromarray( imgs_comb)
         imgs_comb.save(combofile)

         imgs_comb1 = np.hstack( (np.asarray(i) for i in [cur_img, guess_img]))
         imgs_comb1 = Image.fromarray( imgs_comb1)

         imgs_comb2 = np.hstack( (np.asarray(i) for i in [truth_img, predict_img]))
         imgs_comb2 = Image.fromarray( imgs_comb2)
         imgs_comb2.save(combo3file)

         imgs_comb = np.vstack( (np.asarray(i) for i in [imgs_comb1, imgs_comb2]))
         imgs_comb = Image.fromarray( imgs_comb)

         imgs_comb.save(combo2file)

        

         #img = Image.fromarray(jet(predict[0,:,:,0], bytes=True))

         #fig = plt.figure(figsize=(width/300, width/300), dpi=300, frameon=False)
         #m = Basemap(projection='merc',llcrnrlat=17.151,urcrnrlat=29.549,\
         #         llcrnrlon=114.227, urcrnrlon=127.197, resolution='h')
  #
  #
  #       x, y = m(xlons,xlats)
  #       #x, y = m(lon_array, lat_array)
  #       m.contourf(x,y, predict[0,:,:,0], cmap=jet)
  #
  #       # optional image information
  #       m.drawparallels(np.arange(-90.,91.,10.))
  #       m.drawmeridians(np.arange(-180.,181.,10.))
  #       m.drawcoastlines()
  #
  #       #plt.show()
  #       fig.savefig(pfile)
  #       plt.clr()

         #img = toimage(predict[0,:,:,0])
         img.save(predictfile)
    except:
      pass


#CHANGEME -- model to use
#model_name = "cwb_tversky_rms_563603"
#model_name = "cwb_tversky_norm_512_rms_592194"
#model_name = "cwb_block_512_tversky_rms_603554"
#model_name = "cwb_block_lead_512_tversky_rms_861028"
#model_name = "ci_cwb_hvd_bce_rms_e5_1053947"
#model_name = "cwb_block_512_tversky_rms_603554"
#model_name = "cwbci_dice_rms_4_1971078"
#model_name = "seg_nf10_rms_focal_b235_159739"
#model_name = "cwb_nf20_rms_bce_b046_lr8e5_e150_d5_nf32_fixed_resnet_bc4_726151"
#model_name = "cwb_do20_rms_bcedice_b640_lr8e5_e250_d5_nf32_bc6_717694"
#model_name = "cwb_l90_nf10_rms_bcedice_b40_lr8e5_e150_d6_nf32_bc8_724434"
#model_name = "cwb_nf20_rms_tversky_b235_lr8e5_e150_d6_nf32_bc6_743174"
model_name = "cwb_l90_nf20_rms_tversky_b235_lr8e5_e150_d6_nf32_bc6_744242"
model_name = "cwb_l90_nf20_rms_dice_b235_lr8e5_e150_d6_nf32_bc6_752857"

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
