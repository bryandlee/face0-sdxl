from typing import Tuple, List

import cv2
import numpy as np
import mediapipe as mp
from skimage import transform as trans


def open_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_mediapipe_face_detector(model_selection: int = 1, min_detection_confidence: float = 0.5):
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )
    return detector


def detect_largest_face(
    image: np.ndarray,
    detector: mp.solutions.face_detection.FaceDetection,
) -> Tuple[float] | None:
    """Detects the largest face in the image and returns the bounding box. (ccwh)"""
    detections = detector.process(image).detections
    if not detections:
        return None

    detection = max(
        detections,
        key=lambda d: d.location_data.relative_bounding_box.height
        * d.location_data.relative_bounding_box.width,
    )

    center_x = (
        detection.location_data.relative_bounding_box.xmin
        + 0.5 * detection.location_data.relative_bounding_box.width
    )
    center_y = (
        detection.location_data.relative_bounding_box.ymin
        + 0.5 * detection.location_data.relative_bounding_box.height
    )
    width = detection.location_data.relative_bounding_box.width
    height = detection.location_data.relative_bounding_box.height
    return (center_x, center_y, width, height)


def make_image_face_pair(
    image: np.ndarray,
    face_ccwh: Tuple[float],  # absolute pixel coordinates
    face_keypoint5: List[List[float]] | None = None,
    face_size_in_cropped: int = 160,
    face_crop_size: int = 256,
    image_resize_wh: Tuple[int, int] = (256, 256),
    max_face_to_image_ratio: float | None = None,
    get_mask: bool = False,
    face_mask_expansion: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    image_w, image_h = image.shape[1], image.shape[0]

    # Crop face
    face_w, face_h = face_ccwh[2], face_ccwh[3]
    face_size = int(max(face_w, face_h) * face_crop_size / face_size_in_cropped)

    if face_keypoint5 is not None:  # aligned crop
        face = crop_aligned_face(
            image, face_keypoint5, face_size=face_size_in_cropped, image_size=face_crop_size
        )
    else:
        face_left = int(max(0, face_ccwh[0] - face_size // 2))
        face_top = int(max(0, face_ccwh[1] - face_size // 2))
        face_right = int(min(image_w, face_left + face_size))
        face_bottom = int(min(image_h, face_top + face_size))

        crop_x = face_right - face_left
        crop_y = face_bottom - face_top

        if crop_x > crop_y:
            face_left += (crop_x - crop_y) // 2
            face_right = face_left + crop_y
        elif crop_x < crop_y:
            face_top += (crop_y - crop_x) // 2
            face_bottom = face_top + crop_x

        face_size = face_right - face_left
        face = image[face_top:face_bottom, face_left:face_right].copy()
        face = cv2.resize(face, (face_crop_size, face_crop_size), interpolation=cv2.INTER_LANCZOS4)

    # Crop image: crop image around the face with image_aspect_ratio_wh,
    #             and make sure it does not go out of the original image
    if image_w / image_h > image_resize_wh[0] / image_resize_wh[1]:
        crop_w = int(image_h * image_resize_wh[0] / image_resize_wh[1])
        crop_h = image_h
    else:
        crop_w = image_w
        crop_h = int(image_w * image_resize_wh[1] / image_resize_wh[0])
    if max_face_to_image_ratio is not None:
        max_crop_w = int(face_size * max_face_to_image_ratio)
        if crop_w > max_crop_w:
            crop_ratio = max_crop_w / crop_w
            crop_w = max_crop_w
            crop_h = int(crop_h * crop_ratio)
    crop_left = face_ccwh[0] - crop_w // 2
    crop_left = int(min(max(0, crop_left), image_w - crop_w))
    crop_right = crop_left + crop_w
    crop_top = face_ccwh[1] - crop_h // 2
    crop_top = int(min(max(0, crop_top), image_h - crop_h))
    crop_bottom = crop_top + crop_h
    image_crop = image[crop_top:crop_bottom, crop_left:crop_right].copy()
    image_crop = cv2.resize(image_crop, image_resize_wh, interpolation=cv2.INTER_LANCZOS4)

    mask = None
    if get_mask:
        face_mask_size = face_size * face_mask_expansion
        mask_left = int(max(0, face_ccwh[0] - face_mask_size // 2))
        mask_top = int(max(0, face_ccwh[1] - face_mask_size // 2))
        mask_right = int(min(image_w, mask_left + face_mask_size))
        mask_bottom = int(min(image_h, mask_top + face_mask_size))

        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        mask[mask_top:mask_bottom, mask_left:mask_right] = 1
        mask = mask[crop_top:crop_bottom, crop_left:crop_right]
        mask = cv2.resize(mask, image_resize_wh, interpolation=cv2.INTER_NEAREST)

    return image_crop, face, mask


def make_image_face_pair_with_detection(
    image: np.ndarray,
    detector: mp.solutions.face_detection.FaceDetection,
    face_size_in_cropped: int = 160,
    face_crop_size: int = 256,
    image_resize_wh: Tuple[int, int] = (256, 256),
    get_mask: bool = False,
    face_to_image_ratio: float | None = None,
    default_for_no_face: Tuple[float] = (0.5, 0.5, 0.5, 0.5),
):
    face_ccwh = detect_largest_face(image, detector) or default_for_no_face
    return make_image_face_pair(
        image=image,
        face_ccwh=face_ccwh,
        face_size_in_cropped=face_size_in_cropped,
        face_crop_size=face_crop_size,
        image_resize_wh=image_resize_wh,
        get_mask=get_mask,
        max_face_to_image_ratio=face_to_image_ratio,
    )


def make_image_face_pair_with_detection_insightface(
    image: np.ndarray,
    detector,
    **kwargs,
):
    faces = detector.get(image)

    # select largest face
    face = None
    if len(faces) == 0:
        return None, None, None
    elif len(faces) > 1:
        face_areas = [
            (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]) for face in faces
        ]
        face = faces[np.argmax(face_areas)]
    else:
        face = faces[0]

    face_xyxy = face["bbox"].tolist()
    face_w = face_xyxy[2] - face_xyxy[0]
    face_h = face_xyxy[3] - face_xyxy[1]
    face_ccwh = [face_xyxy[0] + face_w / 2, face_xyxy[1] + face_h / 2, face_w, face_h]

    return make_image_face_pair(
        image=image,
        face_ccwh=face_ccwh,
        face_keypoint5=face["kps"].tolist(),
        **kwargs,
    )


def get_insightface_analyzer(name: str = "buffalo_l"):
    import insightface

    detector = insightface.app.FaceAnalysis(name=name, allowed_modules=["detection"])
    detector.prepare(ctx_id=0)
    return detector


# From: insightface
arcface_dst = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def estimate_face_alignment_transfrom(keypoint5, face_size=112, image_size=112):
    if isinstance(keypoint5, list):
        keypoint5 = np.array(keypoint5)
    assert keypoint5.shape == (5, 2)
    ratio = float(face_size) / 112.0
    translate = (image_size - face_size) * 0.5
    dst = arcface_dst * ratio + translate
    tform = trans.SimilarityTransform()
    tform.estimate(keypoint5, dst)
    M = tform.params[0:2, :]
    return M


def crop_aligned_face(image, keypoint5, face_size=112, image_size=112):
    M = estimate_face_alignment_transfrom(keypoint5, face_size, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0)
    return warped
