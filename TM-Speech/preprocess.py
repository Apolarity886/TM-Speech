import argparse
import yaml
from preprocessor.preprocessor import Preprocessor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default='/home/wl/anaconda3/envs/audio/wl_windows/TM-Speech/configs/AISHELL3/preprocess.yaml', help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.configs, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    build_from_path = preprocessor.build_from_path()
