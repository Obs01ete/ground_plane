"""
Ground plane detection with RANSAC
Author: Dmitrii Khizbullin
"""

from typing import Union
import numpy as np
import math
import sys


def plane_by_3_points(points: np.ndarray) -> Union[np.ndarray, None]:
    """ http://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/ """

    p1, p2, p3 = points

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    cp_len = np.linalg.norm(cp)
    if cp_len < 1e-6:
        return None

    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = -np.dot(cp, p3)

    plane_unnorm = np.array((a, b, c, d))

    plane = plane_unnorm / cp_len

    # Normalize the direction of the plane for z to be upwards
    # Actually a plane can be defined by a z-downwards vector in a correct
    # way as well, but the test system seems to require Z to be positive
    if plane is not None:
        if plane[2] < 0.0:
            plane = -plane

    return plane


def load_input(path):
    try:
        with open(path, "r") as f:
            param_p_str = f.readline().rstrip()
        param_p = float(param_p_str)
        points = np.loadtxt(path, dtype=np.float, delimiter='\t', skiprows=2)
    except:
        # print("Error: cannot read the file {}".format(path))
        return None, None
    return param_p, points


def count_inliers(plane, param_p, points):
    normal = plane[0:3]
    d = plane[3]
    dot = np.dot(points, normal)
    mask_left = dot >= d - param_p
    mask_right = dot <= d + param_p
    mask = np.logical_and(mask_left, mask_right)
    num_inliers = np.sum(mask.astype(np.int))
    return num_inliers


"""
  https://en.wikipedia.org/wiki/Random_sample_consensus
"""
def extract_plane(points: np.ndarray, param_p: float):
    if len(points) < 3:
        return None

    inlier_fraction = 0.5
    dimentionality = 3
    assert points.shape[-1] == dimentionality
    num_support_points = 3
    success_prob = 0.99

    num_samples = math.ceil(
        math.log(1.0 - success_prob) / \
        math.log(1 - inlier_fraction ** num_support_points))

    best = None
    for i in range(num_samples):
        indices = np.arange(len(points))
        np.random.shuffle(indices)
        indices_3 = indices[:num_support_points]
        support = np.take(points, indices_3, axis=0)
        hypot_plane = plane_by_3_points(support)
        if hypot_plane is None:
            continue
        num_inliers = count_inliers(hypot_plane, param_p, points)
        if best is None or num_inliers > best['num_inliers']:
            best = {'num_inliers': num_inliers, 'plane': hypot_plane}
        pass

    best_plane = best['plane'] if best is not None else None

    return best_plane


def analyse(case_file_path, result_file_path=None, print_to_stdout=True):
    # Loading input
    param_p, points = load_input(case_file_path)
    if points is None:
        return

    # Running RANSAC
    plane = extract_plane(points, param_p)

    # Saving result
    result_str = " ".join(["{:.6f}".format(v) for v in plane])

    if print_to_stdout:
        print(result_str)
    else:
        if result_file_path is not None:
            with open(result_file_path, "w") as f:
                f.write(result_str + "\n")

    return


def main():
    debug_mode = False

    if debug_mode:
        if True:
            for case_id in range(1, 4):
                analyse("test_cases/case{}.txt".format(case_id))

        if True:
            analyse("sdc_point_cloud.txt", "output.txt")

    else:
        analyse("input.txt", "output.txt")

    pass


if __name__ == "__main__":
    main()
