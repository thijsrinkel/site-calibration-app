import numpy as np
import numpy.typing as npt


def best_fit_transform(
    P_local: npt.NDArray[np.float64], P_world: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Finds R, t such that: P_world ≈ P_local @ R.T + t
    """
    # Step 1: centroids
    cL = P_local.mean(axis=0)
    cW = P_world.mean(axis=0)

    # Step 2: subtract centroids
    X = P_local - cL
    Y = P_world - cW

    # Step 3: compute covariance
    H = X.T @ Y

    # Step 4: SVD
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Step 5: reflection check
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Step 6: translation
    t = cW - cL @ R.T  # or cW - R @ cL depending on convention

    return R, t


def get_rotation_error(
    P_local: npt.NDArray[np.float64],
    P_world: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    P_local_rotated = P_local @ R.T + t

    # error = np.abs(P_local_rotated - P_world)

    return np.linalg.norm(P_local_rotated - P_world, axis=1)


def rotation_matrix_to_rpy(R: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """
    Convert a rotation matrix (3x3) to roll, pitch, yaw (in radians)
    using Qinsy conventions.
    """
    # safety for asin domain
    if R[2, 0] > 1:
        R[2, 0] = 1
    if R[2, 0] < -1:
        R[2, 0] = -1

    roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    pitch = float(np.degrees(-np.arcsin(R[2, 0])))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    qinsy_roll = pitch
    qinsy_pitch = roll
    qinsy_yaw = yaw

    return qinsy_roll, qinsy_pitch, qinsy_yaw


def fit_plane(X):
    """
    Singular value decomposition method.
    Source: https://gist.github.com/lambdalisue/7201028
    """
    # Find the average of points (centroid) along the columns
    C = np.average(X, axis=0)

    # Create CX vector (centroid to point) matrix
    CX = X - C
    # Singular value decomposition
    _, _, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1]

    # Extract a, b, c, d coefficients.
    x0, y0, z0 = C
    a, b, c = N
    d = -(a * x0 + b * y0 + c * z0)

    return a, b, c, d
