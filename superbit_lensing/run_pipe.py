import os
import shutil

from argparse import ArgumentParser

import utils
from pipe import SuperBITPipeline, make_test_config

parser = ArgumentParser()

parser.add_argument('config_file', type=str,
                    help='Config filename')

def main():

    args = parser.parse_args()
    config_file = args.config_file

    config = utils.read_yaml(config_file)
    vb = config['run_options']['vb']

    logfile = 'pipe.log'
    logdir = config['run_options']['outdir']
    log = utils.setup_logger(logfile, logdir=logdir)

    if vb:
        print(f'config =\n{config}')

    pipe = SuperBITPipeline(config_file, log=log)

    rc = pipe.run()

    return rc

if __name__ == '__main__':
    rc = main()

    if rc == 0:
        print('\nScript completed without errors')
    else:
        print(f'\nScript failed with rc={rc}')
