#!/usr/bin/env python3
import os
import random
import click
import yaml

from noisifier import Noisifier, NoiseBank, SUPPORTED_EXTENSIONS

DEFAULT_CONFIG_FILE = 'example.config.yaml'
DEFAULT_OUTPUT_DIR = 'noisifier_output'


@click.command()
@click.argument('audio',
                # help='Path to the audio file or the directory '
                #               'with audio files that need to be noisified',
                type=click.Path(exists=True))
@click.option('--output_dir', '-o', default=DEFAULT_OUTPUT_DIR,
              help='Path to the directory with resulting files',
              type=click.Path())
@click.option('--config_file', '-c', default=DEFAULT_CONFIG_FILE,
              help='Path to the config file',
              type=click.File())
def noisify(audio, output_dir, config_file):
    """
    Adds noise like background noise or beeps to the given audio file(s).
    When the directory is given, it is processed recursively, and
    `output_audio` is regarded as the name of the output directory.
    """

    config = yaml.load(config_file)
    noise_bank = NoiseBank(config)
    noisifier = Noisifier(noise_bank, config)

    if config['random_seed']:
        random.seed(config['random_seed'])

    audio = os.path.abspath(audio)
    output_dir = os.path.abspath(output_dir)
    if os.path.isdir(audio):
        for dirpath, dirnames, filenames in os.walk(audio):
            for file in filenames:
                _, fext = os.path.splitext(file)
                if fext in SUPPORTED_EXTENSIONS:
                    noisifier.process_one_file(
                        audio,
                        os.path.join(dirpath, file),
                        output_dir
                    )
    elif os.path.isfile(audio):
        noisifier.process_one_file(os.path.dirname(audio), audio, output_dir)


if __name__ == '__main__':
    noisify()
