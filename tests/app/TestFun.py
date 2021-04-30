import numpy as np
import quaternion
from numpy.linalg import inv

cam_baseline = 0.2
cam_focalLength = 450
height = 640
width = 960
cx = 480
cy = 320

dic = {0: "move_forward", 1: "turn_left", 2: "turn_right"}

R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
tcyclopean_leftcamera = np.array([-cam_baseline/2, 0, 0])
tleftcamera_cyclopean = np.array([cam_baseline/2, 0, 0])


def targetPixelInCurrentCamera(agent, targetPositionW):
    R_c_s = R_s_c.T

    R_w_s = quaternion.as_rotation_matrix(agent.state.sensor_states["left_sensor"].rotation)

    R_s_w = R_w_s.T
    t_w_s = agent.state.sensor_states["left_sensor"].position
    t_s_w = -1 * R_s_w.dot(t_w_s)

    targetPositionS = R_s_w.dot(targetPositionW) + t_s_w

    targetPositionC = R_c_s.dot(targetPositionS)

    zc = targetPositionC[2]

    uL = (cam_focalLength * targetPositionC[0]) / zc + cx
    v = (cam_focalLength * targetPositionC[1]) / zc + cy


    uR = uL - ((cam_focalLength * cam_baseline) / zc)


    return np.array([uL, uR, v])

def makeJacobian(input):

    xL = input[0]
    xR = input[1]
    y = input[2]

    front_J = cam_baseline / ((xL - xR) ** 2)

    J = np.array([[-xR, xL, 0],
                  [-y, y, xL - xR],
                  [-cam_focalLength, cam_focalLength, 0]])

    J = front_J * J

    return J

def makeQ(depth):
    mvInvLevelSigma2 = np.ones(8)
    mvLevelSigma2 = np.ones(8)
    mvScaleFactor = np.ones(8)
    mvInvScaleFactor = np.ones(8)

    for i in range(1, 8):
        mvScaleFactor[i] = mvScaleFactor[i - 1] * 1.2
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i]

    for i in range(8):
        mvInvScaleFactor[i] = 1.0 / mvScaleFactor[i]
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i]

    level = int(depth * 7 / 10)


    Q = mvLevelSigma2[level] * np.identity(3)

    return Q

def computeObsCovariance(agent, depth_pair, targetPositionW):

    pixels = targetPixelInCurrentCamera(agent, targetPositionW)

    pixels_withoutOffset = np.array([pixels[0] - cx, pixels[1] - cx, pixels[2] - cy])

    J = makeJacobian(pixels_withoutOffset)
    Q = makeQ(depth_pair[int(pixels[2]), int(pixels[0])])

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)

    R = R_w_s.dot(R_s_c)

    U_obs = np.linalg.multi_dot([R, J, Q, J.T, R.T])

    return U_obs

def objectiveFun(U_prior, U_obs):
    U_post = (inv(U_prior) + inv(U_obs))
    U_post = inv(U_post)

    return U_post, U_post[0, 0] + U_post[1, 1] + U_post[2, 2]


def targetPixelInCurrentCamera(agent, targetPositionW):
    R_c_s = R_s_c.T

    R_w_s = quaternion.as_rotation_matrix(agent.state.sensor_states["left_sensor"].rotation)

    R_s_w = R_w_s.T
    t_w_s = agent.state.sensor_states["left_sensor"].position
    t_s_w = -1 * R_s_w.dot(t_w_s)

    targetPositionS = R_s_w.dot(targetPositionW) + t_s_w

    targetPositionC = R_c_s.dot(targetPositionS)

    zc = targetPositionC[2]

    uL = (cam_focalLength * targetPositionC[0]) / zc + cx
    v = (cam_focalLength * targetPositionC[1]) / zc + cy


    uR = uL - ((cam_focalLength * cam_baseline) / zc)


    return np.array([uL, uR, v])
import cv2
def drawFeaturePoints(leftOld, leftIm, uL_gt, v_gt, uL, v):
    cv2.circle(leftIm, (round(uL_gt), round(v_gt)), 5, (0, 0, 255), -1)
    cv2.circle(leftIm, (round(uL), round(v)), 5, (0, 255, 255), -1)