"""yolo v8の学習プログラム."""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_yaml", type=str, help="学習データのyamlファイルパス")
    parser.add_argument(
        "-e", "--epochs", default=20, type=int, help="エポック数 (default:20)")
    parser.add_argument(
        "-w", "--weights", default="yolov8n-seg", type=str, help="追加学習の時の重みファイル.pt指定 (default:yolov8n-seg)")
    return parser.parse_args()


def main(args):
    # Load a model
    model = YOLO(args.weights)

    # Train the model
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=640,
        project="./runs"
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
