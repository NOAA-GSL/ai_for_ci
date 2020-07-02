#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   process_data.py

   prepare data for CWB CI project
   This script only takes satellite bands 8 and 13 as training data and 
   radar data > 35 dBZ as yes/no for convection

   author Jebb Q Stewart (jebb.q.stewart@noaa.gov)
   Copyright 2019. Colorado State University. All rights reserved.

"""

import dateutil.parser
from datetime import timedelta
import numpy as np
import os
import gzip
import multiprocessing 
import argparse
import zarr
import sys
from skimage.transform import resize


import logging

logging_format = '%(asctime)s - %(name)s - %(message)s'
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, 
    format=logging_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("ProcessData")

# radarBase = "/wave/mlp/cwb-ci/Radar/raw/"
# satBase = "/wave/mlp/cwb-ci/Satellite/raw/"
radarBase = "../../data/cwb-ci/Radar/raw/"
satBase = "../../data/cwb-ci/Satellite/raw/"

global compressor
compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
synchronizer = zarr.ProcessSynchronizer('example.sync')

def bilinear_resize(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_l = np.floor(x_ratio * x).astype('int32')
  y_l = np.floor(y_ratio * y).astype('int32')

  x_h = np.ceil(x_ratio * x).astype('int32')
  y_h = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_l
  y_weight = (y_ratio * y) - y_l

  a = image[y_l * img_width + x_l]
  b = image[y_l * img_width + x_h]
  c = image[y_h * img_width + x_l]
  d = image[y_h * img_width + x_h]

  resized = a * (1 - x_weight) * (1 - y_weight) + \
            b * x_weight * (1 - y_weight) + \
            c * y_weight * (1 - x_weight) + \
            d * x_weight * y_weight

  return resized.reshape(height, width)

def readSatellite(file):
  '''
  Read satellite data, binary fortran file with unsigned chars
  '''

  b = np.fromfile(file, dtype=np.uint8)
  if b.shape[0] == 811401:
     b = b.reshape(881,921)
  else:
     logger.error("Error reading satellite file: %s :: %s", b.shape, file)
     b = None

  return b

def readRadar(file):
  '''
  Read radar data, binary fortran file with a bunch of metadata per file
  '''    
  
  radardt = np.dtype({'names': ['yyyy','mm', 'dd', 'hh', 'mn', 'ss', 'nx', 'ny', 'nz', 'proj',
                            'mapscale', 'projlat1', 'projlat2', 'projlon', 'alon', 'alat', 
                            'xy_scale', 'dx', 'dy', 'dxy_scale'], 
                  'formats': [np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
                              np.int32, np.int32, np.uint32, np.int32, np.int32, np.int32, np.int32,
                              np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32   ]})

  if isinstance(file, str) and file.endswith(".gz"):
      logger.debug("reading gzipped file")
      f = gzip.GzipFile(file, 'rb').read()
  else: 
      f = open(file, 'rb').read()

  offset = 0
  data = np.frombuffer(f, dtype=radardt, count=1)
  offset += data.nbytes

  levels = np.frombuffer(f, dtype=np.int32, count=data['nz'][0], offset=offset)
  offset += levels.nbytes
  z_scale = np.frombuffer(f, dtype=np.int32, count=1, offset=offset)
  offset += z_scale.nbytes
  i_bb_mode = np.frombuffer(f, dtype=np.int32, count=1, offset=offset)
  offset += i_bb_mode.nbytes
  unkn01 = np.frombuffer(f, dtype=np.int32, count=9, offset=offset)
  offset += unkn01.nbytes
  varname = np.frombuffer(f, dtype=np.uint8, count=20, offset=offset)
  offset += varname.nbytes
  units = np.frombuffer(f, dtype=np.uint8, count=6, offset=offset)
  offset += units.nbytes
  var_scale = np.frombuffer(f, dtype=np.int32, count=1, offset=offset)
  offset += var_scale.nbytes
  missing = np.frombuffer(f, dtype=np.int32, count=1, offset=offset)
  offset += missing.nbytes
  nradar = np.frombuffer(f, dtype=np.int32, count=1, offset=offset)
  offset += nradar.nbytes

  mosradar = np.frombuffer(f, dtype=np.uint8, count=4*nradar[0], offset=offset)
  offset += mosradar.nbytes

  refl = np.frombuffer(f, dtype=np.int16, offset=offset)
  data = np.frombuffer(f, dtype=radardt, count=1)

  width = data['nx'][0]
  height = data['ny'][0]

  refl = refl.astype(np.float32)/float(var_scale[0])
  refl = refl.reshape(height,width)

  refl[refl< 0] = np.nan

  return refl

def processTimeBlock(validTime, width=512, height=512, steps=3, minutes=10, lead_time=30):
  '''
  processTime
    * validTime is time wanted to process
  Looks for files, if they exist, reads them.  Only create training set if dBZ > 35
  '''
  
  logger.info("Processing for %s", validTime)
  bands = 7
  finalTrain = None
  finalTest = None

  failed = False
  # get data prior to actual time
  for i in range(steps, 0, -1):
      offset = (i-1) * minutes + lead_time
      timestamp = validTime - timedelta(minutes=offset)
      try: 
        train, test, xxx = processTime(timestamp, width=width, height=height)
        if train is None or test is None:
             failed = True
             break
             
        else:
            if finalTrain is None:
    #                 finalTrain = np.zeros((1,512,512, bands*steps), dtype=train.dtype)
                 finalTrain = np.zeros((1,train.shape[1], train.shape[2], bands*steps), dtype=train.dtype)

            index = (i-1) * bands
            for i in range(0, bands):
               #finalTrain[0,:,:,index + i] = np.array(Image.fromarray(train[0,:,:,i]).resize((512, 512), Image.LANCZOS))
               finalTrain[0,:,:,index + i] = train[0,:,:,i]
      except:
        logger.warning("Error reading time block for %s", validTime)
        logger.error(sys.exc_info())
        failed = True
        break

  #        if i == 1:
  #             finalTest = test

  # get current data
  try:
    train, test, xxx = processTime(validTime, width=width, height=height) 
    if test is None:
        failed = True
    else:
        finalTest = test 
  except:
    logger.warning("Error getting current time data for %s", validTime)
    failed = True
      #finalTest = np.array(Image.fromarray(test[0,:,:,0]).resize((512,512), Image.LANCZOS))

  if failed:
       return None, None, validTime
  else: 
   return finalTrain, finalTest, validTime

def processTime(validTime, width=1024, height=1024):
    logger.debug("processing %s", validTime)
    radarFile = validTime.strftime(radarBase + "%Y%m%d/compref_mosaic/COMPREF.%Y%m%d.%H%M.gz")

    bands = ["B08.GSD.Cnt", "B09.GDS.Cnt", "B10.GDS.Cnt", "B11.GDS.Cnt", 
             "B13.GSD.Cnt", "B14.GDS.Cnt", "B16.GSD.Cnt"]

    # b08File = validTime.strftime(satBase+"%Y-%m/%Y-%m-%d_%H%M.B08.GSD.Cnt")
    # b09File = validTime.strftime(satBase+"%Y-%m/%Y-%m-%d_%H%M.B09.GDS.Cnt")
    # b10File = validTime.strftime(satBase+"%Y-%m/%Y-%m-%d_%H%M.B10.GDS.Cnt")
    # b11File = validTime.strftime(satBase+"%Y-%m/%Y-%m-%d_%H%M.B11.GDS.Cnt")
    # b13File = validTime.strftime(satBase+"%Y-%m/%Y-%m-%d_%H%M.B13.GSD.Cnt")
    # b14File = validTime.strftime(satBase+"%Y-%m/%Y-%m-%d_%H%M.B14.GDS.Cnt")
    # b16File = validTime.strftime(satBase+"%Y-%m/%Y-%m-%d_%H%M.B16.GSD.Cnt")

    train = None
    test = None
    year = validTime.strftime("%Y")

    radarData = None
    logger.info ("looking for %s", radarFile)

    if os.path.isfile(radarFile):
      logger.info ("   found, reading data")
      radarData = readRadar(radarFile)
      if np.isnan(radarData).all():
        logger.warn("   Radar data for %s is all NAN", radarFile)

      if np.nanmax(radarData) >= 35 or (radarData>35).sum() > 9:
        logger.info("  max %s :: min %s", np.max(radarData), np.min(radarData)) 
        radarData = np.flip(radarData, axis=0)
        logger.debug("%s minmax: %s :: %s", validTime, np.nanmin(radarData), np.nanmax(radarData))
        radarData[np.isnan(radarData)] = 0
        radarData = radarData[150:-150,150:-150]
        radarData = bilinear_resize(radarData, width, height)
        logger.info("  after max %s :: min %s", np.max(radarData), np.min(radarData)) 

        test = radarData[np.newaxis, :, :, np.newaxis]
      else:
        logger.warn("  data did not exceed thresholds, skipping")      
    else:
      logger.warning("Missing radar file: %s", radarFile)

    train = np.zeros((1, width, height, 7), dtype=np.int32)
    count = 0;
    for b in bands:
      logger.info("processing band: %s", b)
      band_file = validTime.strftime(satBase+"%Y-%m/%Y-%m-%d_%H%M." + b)
      logger.info("looking for %s", band_file)
      if os.path.isfile(band_file):
        logger.info("  reading data")
        tmpdata = readSatellite(band_file)
        logger.info("  verifying data")
        if tmpdata is None:
          raise Exception("Satellite data for " + band_file + " is all NAN")

        logger.info("  processing data")
        logger.info("  max %s :: min %s", np.max(tmpdata), np.min(tmpdata))
        tmpdata = tmpdata[150:-150,150:-150]
        tmpdata = bilinear_resize(tmpdata, width, height).astype(np.int32)
        logger.info("  after resize max %s :: min %s", np.max(tmpdata), np.min(tmpdata))

        train[0,:,:,count] = tmpdata
      else:
        logger.warning("Missing satellite file: %s", band_file)
        raise Exception("Missing satellite file: " + band_file)

      count += 1

    return train, test, validTime

def write_data(result):
    train, test, validTime= result
    if train is not None and test is not None:
        year = validTime.strftime("%Y")
        filename = fullpath

        training_data = None
        test_data = None

        if os.path.isdir(filename +"/train"):
            storage = zarr.open(filename)
            training_data = storage['train']
            test_data = storage['test']
        else:
            store = zarr.DirectoryStore(filename)
            base = zarr.group(store, overwrite=True, synchronizer=synchronizer)

            training_data = base.create_dataset('train', shape=(0,train.shape[1], train.shape[2],train.shape[3]),
                    chunks=(1,train.shape[1], train.shape[2],train.shape[3],), dtype=train.dtype, compressor=compressor)
            test_data = base.create_dataset('test', shape=(0,test.shape[1], test.shape[2],1),
                    chunks=(1,test.shape[1], test.shape[2],), dtype=test.dtype, compressor=compressor)

            # sample code to test images   
            # sat1 = train[0,:,:,0]
            # print (sat1.shape)

            # img = toimage(sat1)
            # img.show()

            # sat2 = train[0,:,:,1]
            # print (sat2.shape)
            # img = toimage(sat2)
            # img.show()

            # lab = test[0,:,:,0]
            # img = toimage(lab)
            # img.show()

        training_data.append(train)
        test_data.append(test)
        print ("AFTER  ", multiprocessing.current_process().name, " :: ", training_data.shape[0], " :: ", test_data.shape[0])
    
def main(outfile, start, end):

    procs = multiprocessing.cpu_count()
    logger.info(" Processors Available: %s", procs)

    procs = 7
    pool = multiprocessing.Pool(processes=max(1, procs-1))


    global fullpath
    fullpath = outfile
    
    #timestamp = "2018-05-20 23:40:00"
    timestamp = start

    validTime = dateutil.parser.parse(timestamp)

    #endTime = dateutil.parser.parse("2017-06-30 23:59:59")
    endstamp = end
    endTime = dateutil.parser.parse(end)

    args = []
    while validTime < endTime:
        
        args.append(validTime)
        validTime = validTime + timedelta(minutes=10)

    for result in pool.imap(processTimeBlock, args):
         write_data(result) 

    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Processor")

    parser.add_argument('-d', '--destination',
       dest="destination", required=True,
       help="name of output file")

    parser.add_argument('-s', '--start',
       dest="start", required=True, default=None,
       help="start iso time")

    parser.add_argument('-e', '--end',
       dest="end", required=True, default=None, type=str,
       help="end iso time")

    args = parser.parse_args()

    outfile = args.destination
    start = args.start
    end = args.end

    main(outfile, start, end)
