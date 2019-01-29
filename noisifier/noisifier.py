import os
import random
import itertools
import click
import librosa
import numpy as np


SUPPORTED_EXTENSIONS = [
    '.wav',
    '.flac',
]


class NoiseBank:
    def __init__(self, config):
        self.config = config
        self.background_noises = []
        self.beep_noises = []

        for noise_path in config['background_noises']:
            NoiseBank.add_noise_path_(noise_path, self.background_noises)
        for noise_path in config['beep_noises']:
            NoiseBank.add_noise_path_(noise_path, self.beep_noises)

        if self.config['verbose']:
            click.echo(f'Found {len(self.background_noises)} background '
                       f'noises and {len(self.beep_noises)} beep noises.')

    def get_random_background_noises(self):
        """
        Returns an iterator over background noises as
        (noise_name, noise) pairs.

        :return: an iterator over background noises
        """
        return map(self.load_noise_,
                   random.sample(self.background_noises,
                                 min(len(self.background_noises),
                                     self.config['n_background_noises'])))

    def get_random_beep_noises(self):
        """
        Returns an iterator over beep noises as
        (noise_name, noise) pairs.

        :return: an iterator over beep noises
        """
        return map(self.load_noise_,
                   random.sample(self.beep_noises,
                                 min(len(self.beep_noises),
                                     self.config['n_beep_noises'])))

    def get_random_noises(self):
        return itertools.chain(
            self.get_random_background_noises(),
            self.get_random_beep_noises()
        )

    @staticmethod
    def add_noise_file_(noise_file, noises):
        fname, fext = os.path.splitext(noise_file)
        if fext in SUPPORTED_EXTENSIONS:
            fname = os.path.basename(fname)
            noises.append((fname, noise_file))

    @staticmethod
    def add_noise_path_(noise_path, noises):
        if os.path.isdir(noise_path):
            for dirpath, dirnames, filenames in os.walk(noise_path):
                for file in filenames:
                    NoiseBank.add_noise_file_(os.path.join(dirpath, file),
                                              noises)
        else:
            NoiseBank.add_noise_file_(noise_path, noises)

    def load_noise_(self, noise_tuple):
        noise_name, noise_file = noise_tuple
        noise, _ = librosa.core.load(noise_file,
                                     sr=self.config['sample_rate'])
        return noise_name, noise


class Noisifier:
    def __init__(self, noise_bank, config):
        self.noise_bank = noise_bank
        self.config = config

    def process_one_file(self, audio_root, file, output_dir):
        """
        Adds some noises from the noise bank to the file.

        :param audio_root: the absolute path to the root directory of the files to be added
        :param file: the absolute path to the input audio file
        :param output_dir: the path to the output directory
        """
        dirpath, fname = os.path.dirname(file), os.path.basename(file)
        fname, fext = os.path.splitext(fname)
        for noise_name, noise in self.noise_bank.get_random_noises():
            self.add_noise(
                file,
                noise,
                os.path.join(
                    output_dir,
                    os.path.relpath(dirpath, audio_root),
                    fname + '.' + noise_name + fext
                )
            )

    def add_noise(self, input_file, noise, output_file):
        if self.config['verbose']:
            click.echo(f'Beep: {input_file} -> {output_file}')
        audio, _ = librosa.core.load(input_file, sr=self.config['sample_rate'])
        noise = np.roll(noise, random.randrange(len(noise)))
        noise = np.tile(noise, reps=(len(audio) - 1) // len(noise) + 1)
        noise = noise[:len(audio)]
        audio += self.config['noise_coeff'] * noise
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        librosa.output.write_wav(
            output_file,
            audio,
            self.config['sample_rate']
        )
