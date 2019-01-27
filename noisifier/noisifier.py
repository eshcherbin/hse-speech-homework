import sys
import os
import click


SUPPORTED_EXTENSIONS = ['.wav', '.flac']
OUTPUT_SUFFIX = '.noise'
DEFAULT_CONFIG_FILE = 'bg_noise.config.yaml'
DEFAULT_OUTPUT_DIR = 'noisifier_output'


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


@click.command()
@click.argument('audio',
                # help='Path to the audio file or the directory '
                #               'with audio files that need to be noisified',
                type=click.Path(exists=True))
@click.option('--output_dir', '-o', default=DEFAULT_OUTPUT_DIR,
              help='Path to the directory with resulting files')
@click.option('--config_file', '-c', default=DEFAULT_CONFIG_FILE,
              help='Path to the config file')
def noisify(audio, output_dir, config_file):
    """
    Adds noise like background noise or beeps to the given audio file(s).
    When the directory is given, it is processed recursively, and
    `output_audio` is regarded as the name of the output directory.

    :param audio: path to the input file or directory
    :param output_dir: path to the directory with the resulting files
    :param config_file: path to the noise config file
    """
    try:
        config = Config.load(config_file)
    except:
        click.echo(f'Problem when loading config from {config_file}', err=True)
        sys.exit(1)

    noise_bank = NoiseBank(config)
    noisifier = Noisifier(noise_bank)

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
