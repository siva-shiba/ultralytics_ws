"""yolo v8の学習プログラム."""

import os
import argparse
import neptune
from ultralytics import YOLO
from ultralytics.utils.callbacks import neptune as neptune_cb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_yaml", type=str, help="学習データのyamlファイルパス")
    parser.add_argument("--project", type=str, required=True,
                        help="Neptuneのプロジェクト名 (例: 'siva-shiba/project-name')")
    parser.add_argument(
        "-e", "--epochs", default=20, type=int, help="エポック数 (default:20)")
    parser.add_argument(
        "-w", "--weights", default="yolov8n-seg", type=str, help="追加学習の時の重みファイル.pt指定 (default:yolov8n-seg)")
    return parser.parse_args()


def get_token(path=".token"):
    """トークンを取得する関数."""
    f = open(path, "r")
    token = f.read()
    f.close()
    # os.environ["NEPTUNE_API_TOKEN"] = "YOUR_REAL_API_TOKEN"
    return token


def main(args):
    # .tokenファイルからAPIトークンを読み込む
    os.environ["NEPTUNE_PROJECT"] = args.project
    os.environ["NEPTUNE_API_TOKEN"] = get_token()

    # Load a model
    model = YOLO(args.weights)

    print(f"Using args: {args}")
    # Train the model
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=640,
        save_period=1,
        project=args.project
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
