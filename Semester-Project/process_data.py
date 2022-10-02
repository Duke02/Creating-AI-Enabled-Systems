import pandas as pd
import numpy as np
import os
import typing as tp
import re


def rotate_z(t: np.ndarray) -> np.ndarray:
    # Rotation about Z axis
    ct: np.ndarray = np.cos(t)
    st: np.ndarray = np.sin(t)
    return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])


def rotate_x(t: np.ndarray) -> np.ndarray:
    ct: np.ndarray = np.cos(t)
    st: np.ndarray = np.sin(t)
    return np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])


def rotate_y(t: np.ndarray) -> np.ndarray:
    ct: np.ndarray = np.cos(t)
    st: np.ndarray = np.sin(t)
    return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])


def eulers_2_rot_matrix(x: np.ndarray) -> np.ndarray:
    # EULER_2_ROT_MATRIX transforms a set of euler angles into a rotation  matrix
    # This might need a couple of 0's attached to it.
    gamma_x: np.ndarray = x[0]
    beta_y: np.ndarray = x[1]
    alpha_z: np.ndarray = x[2]

    return rotate_z(alpha_z) @ rotate_y(beta_y) @ rotate_x(gamma_x)


def movement_name(movement_id: int) -> str:
    # From the original paper (Table 1):
    # https://webpages.uidaho.edu/ui-prmd/Vakanski%20et%20al%20(2018)%20-%20A%20data%20set%20of%20human%20body%20movements%20for%20physical%20rehabilitation%20exercises.pdf
    match movement_id:
        case 1:
            return 'deep_squat'
        case 2:
            return 'hurdle_step'
        case 3:
            return 'inline_lunge'
        case 4:
            return 'side_lunge'
        case 5:
            return 'sit_to_stand'
        case 6:
            return 'standing_active_straight_leg_raise'
        case 7:
            return 'standing_shoulder_abduction'
        case 8:
            return 'standing_shoulder_extension'
        case 9:
            return 'standing_shoulder_internal-external_rotation'
        case 10:
            return 'standing_shoulder_scaption'
        case _:
            return 'ERROR'


def rotate_joint(curr_joint: np.ndarray, prev_joint_angle: np.ndarray, prev_joint_position: np.ndarray) -> np.ndarray:
    rotate_current_joint: np.ndarray = eulers_2_rot_matrix(prev_joint_angle)

    return (rotate_current_joint @ curr_joint.T).T + prev_joint_position[np.newaxis, :]


def smooth(a: np.ndarray, window_size: int):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(window_size, dtype=int), 'valid') / window_size
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(a[:window_size - 1])[::2] / r
    stop = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


if __name__ == "__main__":
    base_data_dir: str = os.path.join('..', 'data', 'SemesterProject')

    # I think I'm gonna work with the Kinect since that one is one I can get a hold of if I need to.

    # Going based on animation.m from the source.

    kinect_pos_base_dir: str = os.path.join(base_data_dir, 'Segmented Movements', 'Kinect', 'Positions')
    kinect_ang_base_dir: str = os.path.join(base_data_dir, 'Segmented Movements', 'Kinect', 'Angles')

    parser_pattern: re.Pattern = re.compile(r'm(?P<movement>\d{2})_s(?P<subject>\d{2})_e(?P<episode>\d{2}).*s\.txt')

    correct_movements: tp.List[tp.Dict[str, tp.Union[np.ndarray, int]]] = []

    """
        % 1 Waist (absolute)
    % 2 Spine
    % 3 Chest
    % 4 Neck
    % 5 Head
    % 6 Head tip
    % 7 Left collar
    % 8 Left upper arm
    % 9 Left forearm
    % 10 Left hand
    % 11 Right collar
    % 12 Right upper arm
    % 13 Right forearm
    % 14 Right hand
    % 15 Left upper leg
    % 16 Left lower leg
    % 17 Left foot
    % 18 Left leg toes
    % 19 Right upper leg
    % 20 Right lower leg
    % 21 Right foot
    % 22 Right leg toes
        """

    joint_connections: np.ndarray = np.array([[4, 6, 5, 3, 2, 3, 7, 8, 9, 3, 11, 12, 13, 1, 15, 16, 17, 1, 19, 20, 21],
                                              [3, 5, 3, 2, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                               22]])

    for fp in os.listdir(kinect_pos_base_dir):
        parsed_match: re.Match = parser_pattern.match(fp)
        movement_id: int = int(parsed_match['movement'])
        subject_id: int = int(parsed_match['subject'])
        episode_num: int = int(parsed_match['episode'])

        print(f'Currently processing movement id {movement_id} / subject id {subject_id} / episode num {episode_num}')

        pos_path: str = os.path.join(kinect_pos_base_dir,
                                     f'm{movement_id:02}_s{subject_id:02}_e{episode_num:02}_positions.txt')
        ang_path: str = os.path.join(kinect_ang_base_dir,
                                     f'm{movement_id:02}_s{subject_id:02}_e{episode_num:02}_angles.txt')

        if not os.path.exists(pos_path) or not os.path.exists(ang_path):
            continue

        pos_array: np.ndarray = np.loadtxt(pos_path, delimiter=',')
        ang_array: np.ndarray = np.loadtxt(ang_path, delimiter=',')

        for i in range(pos_array.shape[1]):
            pos_array[:, i] = smooth(pos_array[:, i], 5)
            ang_array[:, i] = smooth(ang_array[:, i], 21)

        n_frames: int = pos_array.shape[0]

        skeleton_pos: np.ndarray = np.zeros((22, 3, n_frames))
        skeleton_ang: np.ndarray = np.zeros((22, 3, n_frames))
        pos_array1: np.ndarray = pos_array.T
        ang_array1: np.ndarray = ang_array.T

        for i in range(n_frames):
            skeleton_pos[:, :, i] = pos_array1[:, i].reshape((3, 22)).T
            skeleton_ang[:, :, i] = ang_array1[:, i].reshape((3, 22)).T

        skeleton: np.ndarray = np.zeros((22, 3, n_frames))

        for i in range(n_frames):
            joint: np.ndarray = skeleton_pos[:, :, i]
            joint_angle: np.ndarray = np.deg2rad(skeleton_ang[:, :, i])

            # chest, neck, head
            rot_2 = eulers_2_rot_matrix(joint_angle[0, :])
            joint[1, :] = (rot_2 @ joint[1, :].T).T + joint[0, :]
            rot_3 = rot_2 @ eulers_2_rot_matrix(joint_angle[1, :])
            joint[2, :] = (rot_3 @ joint[2, :].T).T + joint[1, :]
            rot_4 = rot_3 @ eulers_2_rot_matrix(joint_angle[2, :])
            joint[3, :] = (rot_4 @ joint[3, :].T).T + joint[2, :]
            rot_5 = rot_4 @ eulers_2_rot_matrix(joint_angle[3, :])
            joint[4, :] = (rot_5 @ joint[4, :].T).T + joint[3, :]
            rot_6 = rot_5 @ eulers_2_rot_matrix(joint_angle[4, :])
            joint[5, :] = (rot_6 @ joint[5, :].T).T + joint[4, :]

            # left arm
            rot_7 = eulers_2_rot_matrix(joint_angle[2, :])
            joint[6, :] = (rot_7 @ joint[6, :].T).T + joint[2, :]
            rot_8 = rot_7 @ eulers_2_rot_matrix(joint_angle[6, :])
            joint[7, :] = (rot_8 @ joint[7, :].T).T + joint[6, :]
            rot_9 = rot_8 @ eulers_2_rot_matrix(joint_angle[7, :])
            joint[8, :] = (rot_9 @ joint[8, :].T).T + joint[7, :]
            rot_10 = rot_9 @ eulers_2_rot_matrix(joint_angle[8, :])
            joint[9, :] = (rot_10 @ joint[9, :].T).T + joint[8, :]

            # right arm
            rot_11 = eulers_2_rot_matrix(joint_angle[2, :])
            joint[10, :] = (rot_11 @ joint[10, :].T).T + joint[2, :]
            rot_12 = rot_11 @ eulers_2_rot_matrix(joint_angle[10, :])
            joint[11, :] = (rot_12 @ joint[11, :].T).T + joint[10, :]
            rot_13 = rot_12 @ eulers_2_rot_matrix(joint_angle[11, :])
            joint[12, :] = (rot_13 @ joint[12, :].T).T + joint[11, :]
            rot_14 = rot_13 @ eulers_2_rot_matrix(joint_angle[12, :])
            joint[13, :] = (rot_14 @ joint[13, :].T).T + joint[12, :]

            # left leg
            rot_15 = eulers_2_rot_matrix(joint_angle[0, :])
            joint[14, :] = (rot_15 @ joint[14, :].T).T + joint[0, :]
            rot_16 = rot_15 @ eulers_2_rot_matrix(joint_angle[14, :])
            joint[15, :] = (rot_16 @ joint[15, :].T).T + joint[14, :]
            rot_17 = rot_16 @ eulers_2_rot_matrix(joint_angle[15, :])
            joint[16, :] = (rot_17 @ joint[16, :].T).T + joint[15, :]
            rot_18 = rot_17 @ eulers_2_rot_matrix(joint_angle[16, :])
            joint[17, :] = (rot_18 @ joint[17, :].T).T + joint[16, :]

            # right leg
            rot_19 = eulers_2_rot_matrix(joint_angle[0, :])
            joint[18, :] = (rot_19 @ joint[18, :].T).T + joint[0, :]
            rot_20 = rot_19 @ eulers_2_rot_matrix(joint_angle[18, :])
            joint[19, :] = (rot_20 @ joint[19, :].T).T + joint[18, :]
            rot_21 = rot_20 @ eulers_2_rot_matrix(joint_angle[19, :])
            joint[20, :] = (rot_21 @ joint[20, :].T).T + joint[19, :]
            rot_22 = rot_21 @ eulers_2_rot_matrix(joint_angle[20, :])
            joint[21, :] = (rot_22 @ joint[21, :].T).T + joint[20, :]

            skeleton[:, :, i] = joint

        correct_movements.append(
            {'movement_id': movement_id, 'subject_id': subject_id, 'episode_num': episode_num, 'positions': pos_array,
             'angles': ang_array, 'positions_path': pos_path, 'angles_path': ang_path, 'skeleton': skeleton})

    df: pd.DataFrame = pd.DataFrame.from_records(correct_movements)

    df['movement_name'] = df['movement_id'].map(movement_name)
    df['reshaped_skeleton'] = df['skeleton'].apply(lambda s: s.reshape((-1, 22, 3)))

    save_path: str = os.path.join('..', 'data', 'SemesterProject', 'processed_data.pkl')
    df.to_pickle(path=save_path)

    print(f'Done! Saved data to "{save_path}"')
