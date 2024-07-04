"""yolo v8の学習プログラム."""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_yaml", type=str, help="学習データのyamlファイルパス")
    return parser.parse_args()


def main(args):
    # Load a model
    model = YOLO("yolov8n-seg.pt")

    # Train the model
    results = model.train(
        data=args.data_yaml,
        epochs=100,
        imgsz=640,
        project="./runs"
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
