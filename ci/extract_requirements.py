"""
Extract requirements from a setup.cfg file; place them into
requirements.txt for the ``install_requires`` and extras_$NAME.txt for
``extras_require``.
"""

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import argparse
import os.path

def make_parser():
    default_setup = os.path.join(os.path.dirname(__file__), "..",
                                 "setup.cfg")
    parser = argparse.ArgumentParser()
    parser.add_argument('setup_cfg', nargs='?', default=default_setup)
    return parser

def main(setup_cfg):
    config = configparser.ConfigParser()
    config.read(setup_cfg)

    reqs = config['options']['install_requires']
    print("requirements.txt")
    with open("requirements.txt", mode='w') as f:
        f.write(reqs)

    for extra, reqs in config['options.extras_require'].items():
        filename = f"extras_{extra}.txt"
        print(filename)
        with open(filename, mode='w') as f:
            f.write(reqs)


if __name__ == "__main__":
    argparser = make_parser()
    args = argparser.parse_args()
    main(args.setup_cfg)
