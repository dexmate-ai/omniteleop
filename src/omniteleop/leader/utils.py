import numpy as np
import pytransform3d.coordinates as coordinates

def project_average_rotation(rot_list: np.ndarray):
    gravity_dir = np.array([0, 0, -1])

    last_mat = rot_list[-1, :, :]
    gravity_quantity = gravity_dir @ last_mat  # (3, )
    max_gravity_axis = np.argmax(np.abs(gravity_quantity))
    same_direction = gravity_quantity[max_gravity_axis] > 0

    next_axis = (max_gravity_axis + 1) % 3
    next_next_axis = (max_gravity_axis + 2) % 3
    angles = []
    for i in range(rot_list.shape[0]):
        next_dir = rot_list[i][:3, next_axis]
        next_dir[2] = 0  # Projection to non gravity direction
        next_dir_angle = coordinates.spherical_from_cartesian(next_dir)[2]
        angles.append(next_dir_angle)

    angle = np.mean(angles)
    final_mat = np.zeros([3, 3])
    final_mat[:3, max_gravity_axis] = gravity_dir * (same_direction * 2 - 1)
    final_mat[:3, next_axis] = [np.cos(angle), np.sin(angle), 0]
    final_mat[:3, next_next_axis] = np.cross(
        final_mat[:3, max_gravity_axis], final_mat[:3, next_axis]
    )
    return final_mat
