#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-02-19
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
from numpy.typing import ArrayLike
import pinocchio as pin
from numpy import array


def normalized(v: ArrayLike) -> np.ndarray:
    """Normalize vector to unit length."""
    v = np.asarray(v)
    return v / np.sqrt(np.sum(v**2))


def rotation_that_transforms_a_to_b(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Find rotation that transform z_start to z_goal, i.e. Rcc_prime."""
    a = normalized(a)
    b = normalized(b)
    v = np.cross(a, b)
    ang = np.arccos(np.dot(a, b))
    if np.abs(ang) < 1e-8:
        return np.eye(3)
    return pin.exp3(ang * normalized(v))


def exponential_function(x: np.ndarray | int | float, a: float, b: float) -> np.ndarray:
    return a * np.exp(-b * x)


def measurement_covariance(
    Tco: np.ndarray, pixel_size: np.ndarray | int | float
) -> np.ndarray:
    return measurement_covariance_prime(Tco, pixel_size)
    # return measurement_covariance_prime_size_independent(Tco, pixel_size)
    # return measurement_covariance_camera_frame(Tco, pixel_size)
    # return measurement_covariance_isotropic_size_dependent(Tco, pixel_size)
    # return measurement_covariance_isotropic_size_independent(Tco, pixel_size)


def measurement_covariance_prime(
    Tco: np.ndarray, pixel_size: np.ndarray | int | float
) -> np.ndarray:
    """Covariance of the measurement."""
    params_xy = array([2.04e-03, 2.42e-06])
    params_z = array([2.06e-02, 5.69e-06])
    params_angle = array([1.24e-01, 4.52e-06])

    var_xy = exponential_function(pixel_size, *params_xy) ** 2
    var_z = exponential_function(pixel_size, *params_z) ** 2
    var_angle = exponential_function(pixel_size, *params_angle) ** 2

    cov_trans_cam_aligned = np.diag([var_xy, var_xy, var_z])
    rot = rotation_that_transforms_a_to_b([0, 0, 1], Tco[:3, 3])
    cov_trans_c = rot @ cov_trans_cam_aligned @ rot.T  # cov[AZ] = A cov[Z] A^T
    rot = Tco[:3, :3].T
    cov_trans_o = rot @ cov_trans_c @ rot.T  # cov[AZ] = A cov[Z] A^T

    cov_o = np.zeros((6, 6))
    cov_o[:3, :3] = np.diag([var_angle] * 3)
    cov_o[3:6, 3:6] = cov_trans_o
    return cov_o

def measurement_covariance_prime_size_independent(
        Tco: np.ndarray, pixel_size: np.ndarray | int | float
) -> np.ndarray:
    std_xy = 0.001089558355755228
    std_z = 0.018276873306908806
    std_rot = 0.11020916669808396
    var_xy = std_xy ** 2
    var_z = std_z ** 2
    var_angle = std_rot ** 2

    cov_trans_cam_aligned = np.diag([var_xy, var_xy, var_z])
    rot = rotation_that_transforms_a_to_b([0, 0, 1], Tco[:3, 3])
    cov_trans_c = rot @ cov_trans_cam_aligned @ rot.T  # cov[AZ] = A cov[Z] A^T
    rot = Tco[:3, :3].T
    cov_trans_o = rot @ cov_trans_c @ rot.T  # cov[AZ] = A cov[Z] A^T

    cov_o = np.zeros((6, 6))
    cov_o[:3, :3] = np.diag([var_angle] * 3)
    cov_o[3:6, 3:6] = cov_trans_o
    return cov_o

def measurement_covariance_camera_frame(
        Tco: np.ndarray, pixel_size: np.ndarray | int | float
) -> np.ndarray:
    params_xy = array([4.52e-03, 2.98e-06])
    params_z = array([1.98e-02, 5.94e-06])
    params_angle = array([1.24e-01, 4.52e-06])

    var_xy = exponential_function(pixel_size, *params_xy) ** 2
    var_z = exponential_function(pixel_size, *params_z) ** 2
    var_angle = exponential_function(pixel_size, *params_angle) ** 2

    cov_trans_c = np.diag([var_xy, var_xy, var_z])
    rot = Tco[:3, :3].T
    cov_trans_o = rot @ cov_trans_c @ rot.T  # cov[AZ] = A cov[Z] A^T

    cov_o = np.zeros((6, 6))
    cov_o[:3, :3] = np.diag([var_angle] * 3)
    cov_o[3:6, 3:6] = cov_trans_o
    return cov_o


def measurement_covariance_isotropic_size_dependent(
        Tco: np.ndarray, pixel_size: np.ndarray | int | float
) -> np.ndarray:
    params_xyz = array([1.20e-02, 5.52e-06])
    params_angle = array([1.24e-01, 4.52e-06])

    var_xyz = exponential_function(pixel_size, *params_xyz) ** 2
    var_angle = exponential_function(pixel_size, *params_angle) ** 2

    cov_trans_c = np.diag([var_xyz, var_xyz, var_xyz])
    rot = Tco[:3, :3].T
    cov_trans_o = rot @ cov_trans_c @ rot.T  # cov[AZ] = A cov[Z] A^T

    cov_o = np.zeros((6, 6))
    cov_o[:3, :3] = np.diag([var_angle] * 3)
    cov_o[3:6, 3:6] = cov_trans_o
    return cov_o


def measurement_covariance_isotropic_size_independent(
        Tco: np.ndarray, pixel_size: np.ndarray | int | float
) -> np.ndarray:
    std_xyz = 0.01067396507824854
    std_rot = 0.11020916669808396
    var_xyz = std_xyz ** 2
    var_angle = std_rot ** 2
    cov_trans_c = np.diag([var_xyz, var_xyz, var_xyz])
    rot = Tco[:3, :3].T
    cov_trans_o = rot @ cov_trans_c @ rot.T  # cov[AZ] = A cov[Z] A^T

    cov_o = np.zeros((6, 6))
    cov_o[:3, :3] = np.diag([var_angle] * 3)
    cov_o[3:6, 3:6] = cov_trans_o
    return cov_o



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T_co = np.eye(4)
    T_co[:3, 3] = [0.2, 0.2, 0.10]
    T_co[:3, :3] = pin.exp3(np.random.rand(3))

    cov = measurement_covariance(T_co, 1000)
    r = np.random.multivariate_normal(np.zeros(6), cov, size=(100,))

    sampled_poses = []
    for v in r:
        T_sampled = np.eye(4)
        T_sampled[:3, :3] = pin.exp3(v[:3])
        T_sampled[:3, 3] = v[3:]
        sampled_poses.append(T_co @ T_sampled)
    sampled_poses = np.asarray(sampled_poses)

    fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
    ax.plot(sampled_poses[:, 0, 3], sampled_poses[:, 2, 3], "o")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    fig.savefig("cov.png")
    plt.show()
