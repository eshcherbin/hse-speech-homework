# Noisifier

Audio data augmentation by adding noise.

### Structure
- `example_audio`: example audio data file
- `example_noises`: example noise files
- `bg_noise.config.yaml`: a config for the bg_noise noise set
- `example.config.yaml`: an example config
- `noisifier.py`: the file with all the important stuff
- `noisify.py`: the executable script

### Config
The configuration file describes the noise dataset as well as some potentially unimportant execution parameters.
It has a self-descriptive easy structure and is written in [YAML](https://yaml.org/):

### Example
The example was tested with Python 3.6, note that you should have the dependencies listed
in the `requirements.txt` (one level above) installed to be able to run the script.

Just run: `python noisify.py -o example_output -c example.config.yaml example_audio`.
You now should have 12 files in the `example_output` directory, 4 per each input file, 
2 of which with background noise and the other 2 with beeps.

To try out a richer noise set, you can download the [bg_noise set](https://yadi.sk/d/ZR5JdkhO3SPoLN).
Then run `python noisify.py -o bg_noise_output -c bg_noise.config.yaml example_audio`.

Finally, run `python noisify.py --help`, you probably won't learn much but it doesn't take much time either 
so just do it and read the info displayed. Or don't, it's completely your choice.

Due to the [CC BY-NC-ND 3.0](https://creativecommons.org/licenses/by-nc-nd/3.0/) license, I simply cannot but tell you 
that one of the example files was taken from [here](https://archive.org/details/SampleVoiceRecordings-KimBorge) 
and is by Kimberly Borge. 
