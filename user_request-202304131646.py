import numpy as np

def rodriguez_rotation_formula(vector, axis, angle):
    """
    Rotate a 3D vector using the Rodriguez rotation formula.

    :param vector: np.ndarray, the 3D vector to be rotated
    :param axis: np.ndarray, the axis of rotation (unit vector)
    :param angle: float, the angle of rotation in radians
    :return: np.ndarray, the rotated 3D vector
    """
    # Ensure that the input vector and axis are numpy arrays
    vector = np.asarray(vector, dtype=float)
    axis = np.asarray(axis, dtype=float)

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Compute the rotated vector using the Rodriguez rotation formula
    rotated_vector = (vector * np.cos(angle) +
                      np.cross(axis, vector) * np.sin(angle) +
                      axis * np.dot(axis, vector) * (1 - np.cos(angle)))

    return rotated_vector

# Example usage:
vector_to_rotate = np.array([1, 0, 0])
rotation_axis = np.array([0, 0, 1])
rotation_angle = np.pi / 2  # 90 degrees in radians

rotated_vector = rodriguez_rotation_formula(vector_to_rotate, rotation_axis, rotation_angle)
print(rotated_vector)  # Should print something like [6.123234e-17, 1.000000e+00, 0.000000e+00]
