import os
import random
import click
import librosa


SUPPORTED_EXTENSIONS = [
    '.wav',
    '.flac',
]


class NoiseBank:
    def __init__(self, config):
        self.config = config
        self.noises = []

        def add_noise(noise_file):
            fname, fext = os.path.splitext(noise_file)
            if fext in SUPPORTED_EXTENSIONS:
                fname = os.path.basename(fname)
                noise = librosa.core.load(noise_file)
                self.noises.append((fname, noise))

        for noise_path in config['noises']:
            if os.path.isdir(noise_path):
                for dirpath, dirnames, filenames in os.walk(noise_path):
                    for file in filenames:
                        add_noise(os.path.join(dirpath, file))
            else:
                add_noise(noise_path)

    def get_noises(self):
        """
        Returns a list of background noises as
        (noise_name, noise) pairs.

        :return: a list of background noises
        """
        return random.sample(self.noises, min(len(self.noises),
                                              self.config['n_noises']))


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
        for noise_name, noise in self.noise_bank.get_noises():
            Noisifier.add_noise(
                file,
                noise,
                os.path.join(
                    output_dir,
                    os.path.relpath(dirpath, audio_root),
                    fname + '.' + noise_name + fext
                )
            )
    @staticmethod
    def add_noise(input_file, noise, output_file):
        click.echo(f'Beep: {input_file} -> {output_file}')
