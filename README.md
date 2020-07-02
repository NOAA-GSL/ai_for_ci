# CI
NOAA GSL Machine Learning Project for the detection of areas likely for convection initiation from satellite imagery

_Developed in partnership with Taiwan Central Weather Bureau_

## Requirements

See requirements.txt

## Running

### Processing data

change the base paths to the location of your data
```
radarBase = "../../data/cwb-ci/Radar/raw/"
satBase = "../../data/cwb-ci/Satellite/raw/"
```

command:
```
process_data.py -d {DESTINATION} -s "{START_TIME_ISO}" -e "{END_TIME_ISO}"
```
example:
```
python -u process_data.py -d ../../data/train -s 2016-05-01T00:00:00 -e 2016-05-01T20:00:00
```

_Note: Best practice is to diversify data for train, validation, and testing from different months, years, etc..._

### Training Model

change the paths to your train, validation, and test data

```  
train_file = '../../data/train'
val_file = '../../data/val'
test_file = '../../data/test'
```

command:
```
keras_segmentation.py -n {NAME_OF_MODEL} -b {LIST_OF_BANDS}
```

There are many other parameters to train the model you can change via the command line.  Please inspect code to understand choices

example:
```
python -u keras_segmentation.py -n ci_model -b 4 6
```

Trained model will be saved as "ci_model.json" in the "models" directory.  Uses bands by array location 4 and 6 for training.

### Test Model

command:
```
test_model.py  -n {NAME_OF_MODEL} -o {IMAGE_OUTPUT_DIRECTORY} -b {LIST_OF_BANDS} -s "{START_TIME_ISO}" -e "{END_TIME_ISO}"
```

example:
```
 python -u test_model.py  -n ci_model -o test -b 4 6 -s 2016-05-02T01:00:00 -e 2016-05-02T03:00:00
```


This will use the model named "ci_model" and save images into a directory "test".  
