import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

skeleton_to_animate: np.ndarray = np.load(os.path.join('..', 'data', 'SemesterProject', 'skeleton_to_animate.npy'))
joint_connections: np.ndarray = np.array([[4, 6, 5, 3, 2, 3, 7, 8, 9, 3, 11, 12, 13, 1, 15, 16, 17, 1, 19, 20, 21],
                                          [3, 5, 3, 2, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]])

fig, ax = plt.subplots()

curr_plot = ax.plot([], [], 'r.')

max_x: float = np.max(skeleton_to_animate[:, :, 2])
min_x: float = np.min(skeleton_to_animate[:, :, 2])

max_y: float = np.max(skeleton_to_animate[:, :, 1])
min_y: float = np.min(skeleton_to_animate[:, :, 1])

# Use this - you're too tired to work right now.
# https://matplotlib.org/stable/gallery/animation/random_walk.html#sphx-glr-gallery-animation-random-walk-py

def init():
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    return ax


# for i in range(skeleton_to_animate.shape[0]):
def animate(frame_idx: int):
    joint: np.ndarray = skeleton_to_animate[frame_idx]
    n_joints: int = joint.shape[0]
    x: np.ndarray = joint[:, 2]
    y: np.ndarray = joint[:, 1]

    ax.scatter(x, y, 'r.', marker_size=15)

    for joint_idx in range(joint_connections.shape[1]):
        point1: float = joint[joint_connections[0, joint_idx], :]
        point2: float = joint[joint_connections[1, joint_idx], :]

        # line([point1(3),point2(3)], [point1(1),point2(1)], [point1(2),point2(2)], 'LineWidth',2);
        ax.line([point1[2], point2[2]], [point1[0], point2[0]], line_width=2)


animation_obj = FuncAnimation(fig, animate, frames=skeleton_to_animate.shape[0], init_func=init, blit=True)

animation_obj.save(os.path.join('..', 'data', 'SemesterProject', 'skeleton.mp4'), fps=5)

plt.show()
