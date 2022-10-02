import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

skeleton_to_animate: np.ndarray = np.load(os.path.join('..', 'data', 'SemesterProject', 'skeleton_to_animate.npy'))
joint_connections: np.ndarray = np.array([[4, 6, 5, 3, 2, 3, 7, 8, 9, 3, 11, 12, 13, 1, 15, 16, 17, 1, 19, 20, 21],
                                          [3, 5, 3, 2, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                           22]]) - 1

fig, ax = plt.subplots()

curr_plot = ax.plot([], [], 'r.')

# What I've tried
# x y
# 2 1 - pretty good
# 0 1
# 1 0
# 1 2
# 0 2
# 2 0
x_idx: int = 2
y_idx: int = 0

max_x: float = np.max(skeleton_to_animate[:, x_idx])
min_x: float = np.min(skeleton_to_animate[:, x_idx])

max_y: float = np.max(skeleton_to_animate[:, y_idx])
min_y: float = np.min(skeleton_to_animate[:, y_idx])


# Use this - you're too tired to work right now.
# https://matplotlib.org/stable/gallery/animation/random_walk.html#sphx-glr-gallery-animation-random-walk-py




def init():
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    return ax.get_children()


# for i in range(skeleton_to_animate.shape[0]):
def animate(frame_idx: int):
    print(f'Doing frame {frame_idx}/{skeleton_to_animate.shape[2]}')

    ax.cla()

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    joint: np.ndarray = skeleton_to_animate[:, :, frame_idx]
    n_joints: int = joint.shape[0]

    x: np.ndarray = joint[:, x_idx]
    y: np.ndarray = joint[:, y_idx]

    ax.scatter(x, y, c='r', s=15, marker='.')

    for joint_idx in range(joint_connections.shape[1]):
        # Don't change this.
        point1: float = joint[joint_connections[1, joint_idx], :]
        point2: float = joint[joint_connections[0, joint_idx], :]

        # line([point1(3),point2(3)], [point1(1),point2(1)], [point1(2),point2(2)], 'LineWidth',2);
        ax.plot([point1[x_idx], point2[x_idx]], [point1[y_idx], point2[y_idx]], linewidth=2)

    return ax.get_children()


total_frames: int = skeleton_to_animate.shape[2]


animation_obj = FuncAnimation(fig, animate, frames=total_frames, init_func=init, blit=True)

save_path: str = os.path.join('..', 'data', 'SemesterProject', f'skeleton_x{x_idx}y{y_idx}.mp4')

animation_obj.save(save_path, fps=15)

plt.show()

print(f'Saved animation to "{save_path}"!')
print('Done!')
