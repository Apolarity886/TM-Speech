import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default='/home/wl/anaconda3/envs/audio/wl_windows/TM-Speech/configs/AISHELL3/preprocess.yaml', help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.configs, "r"), Loader=yaml.FullLoader)
    main(config)
