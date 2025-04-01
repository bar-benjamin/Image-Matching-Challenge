import numpy as np
import csv
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import cv2
import torch
import os
from enum import Enum
import zipfile

# A named tuple containing the intrinsics (calibration matrix K) and
# extrinsics (rotation matrix R, translation vector T) for a given camera.
Gt = namedtuple('Gt', ['K', 'R', 'T'])

# A small epsilon.
eps = 1e-15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src = os.path.join(os.getcwd(), 'input')


class ImageSize(Enum):
    max_img_size_loftr = 840
    max_img_size_quadtree = 1024
    max_img_size_superglue = 1600


class NumPairs(Enum):
    max_num_pairs_loftr = 500
    max_num_pairs_quadtree = 700
    max_num_pairs_superglue = 250


def extract_zip(input_path, output_path):
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    
    os.remove(input_path)


# Taken from provided notebook eval-mteric-and-training-data.ipynb (generalized for Covisibility and Scaling Factors)
def ReadCSVData(filename):
    data_dict = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            data_dict[row[1]] = float(row[2])  # the 1st column is the df index

    return data_dict


# Taken from provided notebook baseline-submission-sift.ipynb
def FlattenMatrix(M, num_digits=8):
    """Convenience function to write CSV files."""

    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


# Taken from provided notebook eval-mteric-and-training-data.ipynb
def LoadCalibration(filename):
    """Load calibration data (ground truth) from the csv file."""

    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue

            camera_id = row[1]
            K = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[3].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[4].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)

    return calib_dict


# Taken from provided notebook eval-mteric-and-training-data.ipynb
def QuaternionFromMatrix(matrix):
    """Transform a rotation matrix into a quaternion."""

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


# Taken from provided notebook eval-mteric-and-training-data.ipynb
def ComputeErrorForOneExample(q_gt, T_gt, q, T, scale):
    """Compute the error metric for a single example.

    The function returns two errors, over rotation and translation.
    These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy."""

    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t


# Taken from provided notebook eval-mteric-and-training-data.ipynb
def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    """Compute the mean Average Accuracy at different thresholds, for one scene."""

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)


# Taken from provided notebook eval-mteric-and-training-data.ipynb (generalized for Covisibility and Scaling Factors)
def DrawMatches(im1, im2, mkpts0, mkpts1, inliers):
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    composite = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    composite[:h1, :w1, :] = im1
    composite[:h2, w1:, :] = im2

    for (p0, p1, inlier) in zip(mkpts0, mkpts1, inliers):
        if inlier > 0:  # Check if the match is an inlier
            # Define a random color for both the line and circles
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

            # Draw line connecting matched points
            cv2.line(composite, (int(p0[0]), int(p0[1])), (int(w1 + p1[0]), int(p1[1])), (0, 255, 0), 1)

            # Draw circles at the matched points
            cv2.circle(composite, (int(p0[0]), int(p0[1])), 5, color, 2)
            cv2.circle(composite, (int(w1 + p1[0]), int(p1[1])), 5, color, 2)

    # Convert composite image from BGR to RGB for matplotlib
    composite = composite[:, :, ::-1]

    plt.figure(figsize=(15, 10))
    plt.imshow(composite)
    plt.show()


# Taken from provided notebook eval-mteric-and-training-data.ipynb
def ComputeFundamentalMatrix(mkpts0, mkpts1):
    MAX_RETRIES = 5
    MAGSAC_threshold = 0.2
    MAGSAC_confidence = 0.99999
    MAGSAC_max_iterations = 10000

    if len(mkpts0) > 8:
        retries = 0
        F, mask = None, None

        while retries < MAX_RETRIES:
            try:
                F, mask = cv2.findFundamentalMat(
                    mkpts0, mkpts1, cv2.USAC_MAGSAC,
                    MAGSAC_threshold, MAGSAC_confidence, MAGSAC_max_iterations
                )
                if F is not None and F.size != 0:
                    inliers = mask.flatten().astype(np.uint8)
                    return F, inliers
            except cv2.error:
                pass
            
            retries += 1

        F = np.zeros((3, 3))
        inliers = np.zeros(mkpts0.shape[0], dtype=np.uint8)

    else:
        F = np.zeros((3, 3))
        inliers = np.zeros(mkpts0.shape[0], dtype=np.uint8)

    return F, inliers


# Took inspiration from openCV C++ open source GitHub repository (cv::decomposeEssentialMat)
def DecomposeFundamentalMatrix(F, K1, K2):
    """Decompose fundamental matrix F into R and T, given the intrinsics calibration matrices."""

    # Essential -> Fundamental matrix conversion
    # formula from: https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)
    E = K2.T @ F @ K1

    # SVD of Essential Matrix
    # formula from: https://en.wikipedia.org/wiki/Essential_matrix#Finding_one_solution
    U, S, Vt = np.linalg.svd(E)

    # proper orientation of U and Vt
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # pre-defined W matrix used in decomposition
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # calculate two possible rotations and the translation
    R1 = U @ W.T @ Vt
    R2 = U @ W @ Vt
    T = U[:, 2] # 3rd column represents direction of translation

    return R1, R2, T


# helper function
def load_predictions(submission_file):
    predictions = {}
    with open(submission_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header
            if i == 0:
                continue
            predictions[row[0]] = np.array([float(v) for v in row[1].split(' ')]).reshape([3, 3])

    return predictions


# Taken from provided notebook eval-mteric-and-training-data.ipynb
# Iterate over all the scenes now. We compute the metric for each scene, and then average it over all scenes.
def EvaluatePredVSGroundTruth(submission_file):
    filename = os.path.join(src, 'train', 'scaling_factors.csv')
    scaling_dict = ReadCSVData(filename)

    # We use two different sets of thresholds over rotation and translation.
    # DO NOT CHANGE THIS -- these are the values used by the competition scoring back-end.
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)

    predictions = load_predictions(submission_file)

    scenes = []
    for pred in predictions.keys():
        scene = pred.split(';')[0]
        if scene not in scenes:
            scenes.append(scene)

    calib_dict = {}
    for scene in scenes:
        filename = os.path.join(src, 'train', scene, 'calibration.csv')
        calib_dict[scene] = LoadCalibration(filename)

    # Save the per-sample errors and the accumulated metric to dictionaries, for later inspection.
    errors = {scene: {} for scene in scenes}
    mAA = {scene: {} for scene in scenes} # mean average accuracy per scene

    for pred, F in tqdm(predictions.items()):
        scene, pair = pred.split(';')
        image_1_id, image_2_id = pair.split('-')

        K1, R1_gt, T1_gt = (calib_dict[scene][image_1_id].K,
                            calib_dict[scene][image_1_id].R,
                            calib_dict[scene][image_1_id].T.reshape((3, 1)))
        K2, R2_gt, T2_gt = (calib_dict[scene][image_2_id].K,
                            calib_dict[scene][image_2_id].R,
                            calib_dict[scene][image_2_id].T.reshape((3, 1)))

        R1, R2, T = DecomposeFundamentalMatrix(F, K1, K2)
        q1 = QuaternionFromMatrix(R1)
        q2 = QuaternionFromMatrix(R2)

        # Get the relative rotation and translation between these two cameras
        dR_gt = np.dot(R2_gt, R1_gt.T)
        dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
        q_gt = QuaternionFromMatrix(dR_gt)
        q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

        # Compute the error
        err_q1, err_t1 = ComputeErrorForOneExample(q_gt, dT_gt, q1, T, scaling_dict[scene])
        err_q2, err_t2 = ComputeErrorForOneExample(q_gt, dT_gt, q2, T, scaling_dict[scene])

        assert err_t1 == err_t2
        errors[scene][pair] = [min(err_q1, err_q2), err_t1]

    print('------- SUMMARY -------')
    for scene in scenes:
        mAA[scene], _, _, _ = ComputeMaa([v[0] for v in errors[scene].values()],
                                         [v[1] for v in errors[scene].values()],
                                         thresholds_q,
                                         thresholds_t)
        print(f'Mean average Accuracy on "{scene}": {mAA[scene]:.05f}')

    print(f'\nMean average Accuracy on dataset: {np.mean(list(mAA.values())):.05f}')


# helper function
def image_to_tensor(image, keepdim = True):
    input_shape = image.shape
    tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(f"Cannot process image with shape {input_shape}")

    return tensor.unsqueeze(0) if not keepdim else tensor


# helper function
def bgr_to_rgb(image):
    return image.flip(-3)


# helper function
def rgb_to_grayscale(image):
    # 8 bit images
    if image.dtype == torch.uint8:
        rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
    # floating point images
    elif image.dtype in (torch.float16, torch.float32, torch.float64):
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
    else:
        raise TypeError(f"Unknown data type: {image.dtype}")

    # unpack the color image channels with RGB order
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b


# helper function
def load_torch_image(img_path, max_img_size, squaring=False):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    scale = max_img_size / max(h, w)
    updated_h, updated_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (updated_w, updated_h))

    if squaring:
        h, w = max(h, w), max(h, w)
        squared_img = np.zeros((max_img_size, max_img_size, 3)).astype(np.uint8)
        squared_img[:updated_h, :updated_w, :] = img
        img = squared_img

    img = image_to_tensor(img, False).float() / 255.
    img = bgr_to_rgb(img)
    return img.to(device), w, h


# helper function
def seed_torch(seed=2025):
    random.seed(seed)
    os.environ['SEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_and_preprocess_images(img_path0, img_path1, max_img_size, squaring):
    img0, w1, h1 = load_torch_image(img_path0, max_img_size=max_img_size, squaring=squaring)
    img1, w2, h2 = load_torch_image(img_path1, max_img_size=max_img_size, squaring=squaring)

    batch = {"image0": rgb_to_grayscale(img0),
             "image1": rgb_to_grayscale(img1)}

    return batch, (w1, h1), (w2, h2), img0, img1


def process_images(mkpts0, mkpts1, mconf, w1, h1, w2, h2, img0, img1, max_num_pairs):
    # select top matches with the least noise
    sorted_idx = np.argsort(-mconf)
    if len(mconf) > max_num_pairs:
        mkpts0 = mkpts0[sorted_idx[:max_num_pairs], :]
        mkpts1 = mkpts1[sorted_idx[:max_num_pairs], :]

    img0_batch_size, img0_h, img0_w, img0_d = img0.shape
    img1_batch_size, img1_h, img1_w, img1_d = img1.shape

    mkpts0[:, 0] = mkpts0[:, 0] * w1 / img0_d
    mkpts0[:, 1] = mkpts0[:, 1] * h1 / img0_w

    mkpts1[:, 0] = mkpts1[:, 0] * w2 / img1_d
    mkpts1[:, 1] = mkpts1[:, 1] * h2 / img1_w

    return mkpts0, mkpts1
