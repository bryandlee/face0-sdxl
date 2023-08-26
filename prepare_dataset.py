import os
import math
import json
from typing import Dict

import fire
import numpy as np
from PIL import Image
from tqdm import tqdm


def prepare_annotations(
    annotation_path: str = "../data/celeba/list_landmarks_celeba.txt",
    image_dir: str = "../data/celeba/images",
    output_dir: str = "../data/celeba/data.json",
    name: str = "celeba",
    min_w: int = 480,
    min_h: int = 480,
    min_face_w: int = 128,
    min_face_h: int = 128,
):
    if name == "celeba":
        annotations = parse_celeba_annotations(annotation_path, image_dir)
    elif name == "ffhq":
        annotations = parse_ffhq_annotations(annotation_path, image_dir)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    print(f"Total number of images: {len(annotations)}")

    num_image_with_face = sum(map(lambda x: len(x["detections"]) > 0, annotations.values()))
    print(
        f"Number of images with face: {num_image_with_face} ({num_image_with_face / len(annotations) * 100:.2f}%)"
    )

    num_image_with_multiple_faces = sum(map(lambda x: len(x["detections"]) > 1, annotations.values()))
    print(
        f"Number of images with multiple faces: {num_image_with_multiple_faces} ({num_image_with_multiple_faces / len(annotations) * 100:.2f}%)"
    )

    largest_face_sizes = map(get_largest_face_size, annotations.values())
    largest_face_sizes = [x for x in largest_face_sizes if x is not None]
    print(f"Average largest face size: {sum(largest_face_sizes) / len(largest_face_sizes):.1f}")

    face_to_image_ratios = map(face_to_image_ratio, annotations.values())
    face_to_image_ratios = [x for x in face_to_image_ratios if x is not None]
    print(f"Average face to image ratio: {sum(face_to_image_ratios) / len(face_to_image_ratios):.3f}")

    filtered_annotations = {
        k: v for k, v in annotations.items() if filter_out(v, min_w, min_h, min_face_w, min_face_h)
    }
    print(
        f"Number of images after filtering: {len(filtered_annotations)} ({len(filtered_annotations) / len(annotations) * 100:.2f}%)"
    )

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(filtered_annotations, f, indent=2, sort_keys=False)

    print(f"Saved to {output_dir}")


def parse_ffhq_annotations(
    annotation_path: str = "../data/ffhq/ffhq-dataset-v2.json",
    image_dir: str = "../data/ffhq/images",
):
    with open(annotation_path, "r") as f:
        original_annotations = json.load(f)
    annotations = {}

    for info in tqdm(original_annotations.values(), desc="Parsing annotations"):
        w, h = info["in_the_wild"]["pixel_size"]
        bbox = info["in_the_wild"]["face_rect"]
        landmarks = info["in_the_wild"]["face_landmarks"]

        image_path = os.path.join(image_dir, os.path.basename(info["in_the_wild"]["file_path"]))

        left_eye = np.mean(np.array(landmarks[36:42]), axis=0).tolist()
        right_eye = np.mean(np.array(landmarks[42:48]), axis=0).tolist()
        nose = np.mean(np.array(landmarks[31:35]), axis=0).tolist()
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]

        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        face_x = bbox[0] + face_w / 2
        face_y = bbox[1] + face_h / 2

        annotations[image_path] = {
            "w": w,
            "h": h,
            "detections": [
                {
                    "bbox": [face_x, face_y, face_w, face_h],
                    "kps": [left_eye, right_eye, nose, left_mouth, right_mouth],
                    "det_score": 1.0,
                }
            ],
        }
    return annotations


def parse_celeba_annotations(
    annotation_path: str = "../data/celeba/list_landmarks_celeba.txt",
    image_dir: str = "../data/celeba/images",
) -> Dict:
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines][2:]

    annotations = {}

    for line in tqdm(lines, desc="Parsing annotations"):
        filename, *landmarks = line.split(" ")
        landmarks = [point for point in landmarks if point != ""]
        landmarks = np.array(landmarks, dtype=np.float32).reshape(-1, 2)

        face_xywh = [
            landmarks[:, 0].min(),
            landmarks[:, 1].min(),
            landmarks[:, 0].max(),
            landmarks[:, 1].max(),
        ]
        face_ccwh = [
            (face_xywh[0] + face_xywh[2]) / 2,
            (face_xywh[1] + face_xywh[3]) / 2,
            face_xywh[2] - face_xywh[0],
            face_xywh[3] - face_xywh[1],
        ]
        face_ccwh[2] *= 2
        face_ccwh[3] *= 2

        face_xywh = [
            face_ccwh[0] - face_ccwh[2] / 2,
            face_ccwh[1] - face_ccwh[3] / 2,
            face_ccwh[0] + face_ccwh[2] / 2,
            face_ccwh[1] + face_ccwh[3] / 2,
        ]

        image_path = os.path.join(image_dir, filename)
        w, h = Image.open(image_path).size

        annotation = {
            "w": w,
            "h": h,
            "detections": [
                {
                    "bbox": face_ccwh,
                    "kps": landmarks.tolist(),
                    "det_score": 1.0,
                }
            ],
        }

        annotations[image_path] = annotation

    return annotations


def get_largest_face_size(annotation):
    faces = annotation["detections"]
    if len(faces) == 0:
        return None
    return max(map(lambda x: max(x["bbox"][2], x["bbox"][3]), faces))


def face_to_image_ratio(annotation):
    w = annotation["w"]
    h = annotation["h"]
    faces = annotation["detections"]
    if len(faces) == 0:
        return None
    return math.sqrt(sum(map(lambda x: x["bbox"][2] * x["bbox"][3], faces)) / (w * h))


def filter_out(
    annotation: Dict, min_w: int = 480, min_h: int = 480, min_face_w: int = 128, min_face_h: int = 128
):
    if len(annotation["detections"]) != 1:
        return False

    w = annotation["w"]
    h = annotation["h"]

    if w < min_w or h < min_h:
        return False

    face = annotation["detections"][0]

    face_w = face["bbox"][2]
    face_h = face["bbox"][3]

    if face_w < min_face_w or face_h < min_face_h:
        return False

    return True


if __name__ == "__main__":
    fire.Fire(prepare_annotations)
