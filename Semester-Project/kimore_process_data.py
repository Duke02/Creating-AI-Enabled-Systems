import numpy as np
import os
import pandas as pd
import typing as tp

# TOOD: Remember to subtract by 1 for each of the indexes when you actually use them
INDEX_SPINE_BASE: int = 1
INDEX_SPINE_MID: int = 5
INDEX_NECK: int = 9
INDEX_HEAD: int = 13
INDEX_SHOULDER_LEFT: int = 17
INDEX_ELBOW_LEFT: int = 21
INDEX_WRIST_LEFT: int = 25
INDEX_HAND_LEFT: int = 29
INDEX_SHOULDER_RIGHT: int = 33
INDEX_ELBOW_RIGHT: int = 37
INDEX_WRIST_RIGHT: int = 41
INDEX_HAND_RIGHT: int = 45
INDEX_HIP_LEFT: int = 49
INDEX_KNEE_LEFT: int = 53
INDEX_ANKLE_LEFT: int = 57
INDEX_FOOT_LEFT: int = 61
INDEX_HIP_RIGHT: int = 65
INDEX_KNEE_RIGHT: int = 69
INDEX_ANKLE_RIGHT: int = 73
INDEX_FOOT_RIGHT: int = 77
INDEX_SPINE_SHOULDER: int = 81
INDEX_TIP_LEFT: int = 85
INDEX_THUMB_LEFT: int = 89
INDEX_TIP_RIGHT: int = 93


def construct_positions(joint_positions: np.ndarray) -> tp.Dict[str, np.ndarray]:
    output: tp.Dict[str, np.ndarray] = dict()

    body_part_to_index: tp.Dict[str, int] = {
        'hip_right': INDEX_HIP_RIGHT,
        'knee_right': INDEX_KNEE_RIGHT,
        'ankle_right': INDEX_ANKLE_RIGHT,
        'foot_right': INDEX_FOOT_RIGHT,
        'hip_left': INDEX_HIP_LEFT,
        'knee_left': INDEX_KNEE_LEFT,
        'ankle_left': INDEX_ANKLE_LEFT,
        'foot_left': INDEX_FOOT_LEFT,
        'spine_base': INDEX_SPINE_BASE,
        'spine_mid': INDEX_SPINE_MID,
        'head': INDEX_HEAD,
        'spine_shoulder': INDEX_SPINE_SHOULDER,
        'shoulder_right': INDEX_SHOULDER_RIGHT,
        'shoulder_left': INDEX_SHOULDER_LEFT,
        'elbow_right': INDEX_ELBOW_RIGHT,
        'wrist_right': INDEX_WRIST_RIGHT,
        'hand_right': INDEX_HAND_RIGHT,
        'elbow_left': INDEX_ELBOW_LEFT,
        'wrist_left': INDEX_WRIST_LEFT,
        'hand_left': INDEX_HAND_LEFT
    }

    for out_key, index in body_part_to_index.items():
        output[out_key] = np.hstack([joint_positions[:, index - 1],
                                     joint_positions[:, index],
                                     joint_positions[:, index + 1]])

    return output


def feature_extraction()
    pass