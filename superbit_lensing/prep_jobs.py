from argparse import ArgumentParser
from jobs import JobsManager

parser = ArgumentParser()

parser.add_argument('jobs_config', type=str,
                    help='Filepath to yaml configuration file for all jobs')
parser.add_argument('--fresh', action='store_true', default=False,
                    help='Clean test directory of old outputs')

def main(args):
    config = args.jobs_config
    fresh = args.fresh

    manager = JobsManager(config, fresh=fresh)
    manager.run()

    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)

    if rc == 0:
        print('\nScript completed without errors')
    else:
        print(f'\nScript failed with rc={rc}')
