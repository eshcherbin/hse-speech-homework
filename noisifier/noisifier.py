import os
import click


class Config:
    def __init__(self):
        print('Config created')

    @staticmethod
    def load(config_file):
        """
        Loads config from the given file.

        :param config_file: path to the config file.
        :return: loaded config
        """
        print(f'Config (almost) loaded from {config_file}')
        return Config()


class NoiseBank:
    def __init__(self, config):
        self.config = config

    def get_background_noises(self):
        """
        Returns an iterator over background noises as
        (noise_name, noise) pairs.

        :return: an iterator over background noises
        """
        return iter([
            ('bg_noise1', None),
            ('bg_noise2', None),
        ])

    def get_beep_noises(self):
        """
        Returns an iterator over beep noises as
        (noise_name, noise) pairs.

        :return: an iterator over background noises
        """
        return iter([
            ('beep_noise1', None),
        ])


class Noisifier:
    def __init__(self, noise_bank):
        self.noise_bank = noise_bank

    def process_one_file(self, audio_root, file, output_dir):
        """
        Adds some noises from the noise bank to the file.

        :param audio_root: the absolute path to the root directory of the files to be added
        :param file: the absolute path to the input audio file
        :param output_dir: the path to the output directory
        """
        dirpath, fname = os.path.dirname(file), os.path.basename(file)
        fname, fext = os.path.splitext(fname)
        for noise_name, noise in self.noise_bank.get_background_noises():
            Noisifier.add_background_noise(
                file,
                noise,
                os.path.join(
                    output_dir,
                    os.path.relpath(dirpath, audio_root),
                    fname + '.' + noise_name + fext
                )
            )
        for noise_name, noise in self.noise_bank.get_beep_noises():
            Noisifier.add_beep_noise(
                file,
                noise,
                os.path.join(
                    output_dir,
                    os.path.relpath(dirpath, audio_root),
                    fname + '.' + noise_name + fext
                )
            )

    @staticmethod
    def add_background_noise(input_file, noise, output_file):
        click.echo(f'Blop: {input_file} -> {output_file}')

    @staticmethod
    def add_beep_noise(input_file, noise, output_file):
        click.echo(f'Beep: {input_file} -> {output_file}')
