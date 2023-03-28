import cv2
import argparse
from tqdm import tqdm
import os
from os import path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="input dir")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="output dir")
    parser.add_argument("--cascade-file", type=str, help="optional cascade classifier (human face detector by default)")
    parser.add_argument("--min-size", type=int, default=320, help="minimum image size")
    parser.add_argument("--max-size", type=int, default=640, help="maximum image size. If the face image is larger than max-size, resize it to max-size.")
    parser.add_argument("--margin", type=float, nargs="+", default=[0.25, 0.5, 0.8], help="face margins")
    parser.add_argument("--prefix", type=str, help="filename prefix")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.cascade_file is not None:
        classifier = cv2.CascadeClassifier(args.cascade_file)
    else:
        classifier = cv2.CascadeClassifier(path.join(cv2.data.haarcascades,
                                                     "haarcascade_frontalface_alt2.xml"))

    for filename in tqdm(os.listdir(args.input_dir), ncols=80):
        if not path.splitext(filename)[-1].lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        filename = path.join(args.input_dir, filename)
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        if args.prefix is not None:
            basename = args.prefix + "_" + path.splitext(path.basename(filename))[0]
        else:
            basename = path.splitext(path.basename(filename))[0]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
        for (x, y, w, h) in faces:
            for margin in args.margin:
                margin = int(max(w, h) * margin)
                margin_top = int(margin * 0.9)
                fx = max(x - margin, 0)
                fy = max(y - margin_top, 0)
                fw = w + margin * 2
                fh = h + margin * 2
                if min(fw, fh) < args.min_size:
                    continue
                face = im[fy:fy + fh, fx:fx + fw]
                max_size = max(face.shape[0], face.shape[1])
                if max_size > args.max_size:
                    scale = args.max_size / max_size
                    face = cv2.resize(face, (int(face.shape[1] * scale), int(face.shape[0] * scale)),
                                      interpolation=cv2.INTER_AREA)
                output_file = path.join(args.output_dir, f"{basename}_{fx}_{fy}_{fw}_{fh}.png")
                cv2.imwrite(output_file, face)


if __name__ == "__main__":
    main()
