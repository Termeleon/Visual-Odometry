import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from UndistortImage import UndistortImage
from ReadCameraModel import ReadCameraModel

# 3.1
fx, fy, cx, cy, _, LUT = ReadCameraModel('./Oxford_dataset_reduced/model')
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 3.2
# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
folder = './Oxford_dataset_reduced/images'
images = []
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename), flags=-1)
    color_img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistorted_img = UndistortImage(color_img, LUT)
    if undistorted_img is not None:
        images.append(undistorted_img)

images = np.array(images)
# 3.3
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# Code copied and modified from https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
sift = cv2.SIFT_create()
rotations = []
translates = []


def match_key_points(i):
    kp1, des1 = sift.detectAndCompute(images[i], None)
    kp2, des2 = sift.detectAndCompute(images[i + 1], None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return pts1, pts2


def find_fundamental_matrix(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return F, mask, pts1, pts2


for i in range(376):
    pts1, pts2 = match_key_points(i)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask, pts1, pts2 = find_fundamental_matrix(pts1, pts2)

    # E = KT F K
    E = K.T @ F @ K

    # 3.6
    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0
    _, R, T, _ = cv2.recoverPose(E, pts1, pts2, K)
    rotations.append(R)
    translates.append(T)

rt = []
prev = np.eye(4)
ori = np.array([[0, 0, 0, 1]]).T
for i in range(376):
    temp = np.concatenate((rotations[i], translates[i]), axis=1)
    temp = np.concatenate((temp, [[0, 0, 0, 1]]), axis=0)
    temp = np.linalg.inv(temp)
    prev = prev @ temp
    rt.append(prev @ ori)
rt = np.array(rt)

x = []
y = []
z = []
for i in rt:
    temp = i[0:3]
    x.append(temp[0, 0])
    y.append(temp[1, 0])
    z.append(temp[2, 0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()

plt.plot(x, z)
plt.show()
