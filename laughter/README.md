## Laughter detection tool

This tool detects laughter interval in audio files.

### Repo structure
`laughter_classification/` - everything related with frame-wise 
 laughter classification: model training, visualization, tuning,
  cross-validation

`laughter_prediction/` - module for laughter prediction for
arbitrary audio file in .wav format

`models/` - serialized pre-learned models for classification

`data/` - pre-extracted features in .csv format

`params/` - configuration files for laughter prediction

`hw.ipynb` - model training and analysis log

### Data
Audio corpus available at 
http://www.dcs.gla.ac.uk/vincia/?p=378 (vocalizationcorpus.zip)

### How to use
usage: process_audio.py [-h] [--wav_path WAV_PATH] [--params PARAMS]

Script for prediction laughter intervals for .wav file

optional arguments:
  -h, --help           show this help message and exit
  --wav_path WAV_PATH  Path to .wav file
  --params PARAMS      /JSON file with the classification parameters. Default:
                       ../params/default_params.json.
