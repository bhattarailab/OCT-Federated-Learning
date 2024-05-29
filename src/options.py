import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default.yaml', nargs='*', 
                        help='Config file name')

    args = parser.parse_args()
    return args