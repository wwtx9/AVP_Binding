import numpy as np
from numpy.linalg import inv
import quaternion
import math

cam_baseline = 0.2
cam_focalLength = 450
height = 640
width = 960
cx = 480
cy = 320

R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
tcyclopean_leftcamera = np.array([-cam_baseline/2, 0, 0])
tleftcamera_cyclopean = np.array([cam_baseline/2, 0, 0])

#Next Best View
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

#Compute observation covarience matrix for all visible keypoints
def computeObsCovarianceMatrixForKeyPoints(CurrentFrame, depth_img):
    N = CurrentFrame.N

    for i in range(N):
            uL, v = CurrentFrame.Kps[i]
            uR, _ = CurrentFrame.KpRights[i]

            # level = CurrentFrame.mKps[i].octave
            depth = depth_img[round(v), round(uL)]

            level = round(depth * 7 / 10)

            #cyclopean coordinate
            xL = uL - cx
            xR = uR - cx
            y = v - cy

            input = np.array([xL, xR, y])
            J = makeJacobian(input)
            Q = makeQ(level)
            Rwc = CurrentFrame.getRotation()

            U_obs = np.linalg.multi_dot([Rwc, J, Q, J.T, Rwc.T])

            CurrentFrame.mObsMatrices_array[i] = U_obs


#Update covarince matrix for all visible keypoints
def computePostCovarianceMatrixForKeyPoints(LastFrame, CurrentFrame):
    N = CurrentFrame.N
    for i in range(N):
        U_obs = CurrentFrame.mObsMatrices_array[i]

        if CurrentFrame.mId == 0:  # fist image
            U_post = (inv(U_obs) + inv(U_obs))
            U_post = inv(U_post)
            CurrentFrame.mPostMatrices_array[i] = U_post
        else:
            # This is a tracked keypoint
            if i < len(CurrentFrame.preTrackedKpIndexs):
                preTrackKpIndex = CurrentFrame.preTrackedKpIndexs[i]
                U_prior = LastFrame.mPostMatrices_array[preTrackKpIndex]
                U_post = (inv(U_obs) + inv(U_prior))
                U_post = inv(U_post)
                CurrentFrame.mPostMatrices_array[i] = U_post
            else:
                # this a new keypoint
                print("Error now  this a new keypoint")
                U_post = (inv(U_obs) + inv(U_obs))
                U_post = inv(U_post)
                CurrentFrame.mPostMatrices_array[i] = U_post


#select active keypoint
def selectActiveKeyPoint(CurrentFrame):
    N = CurrentFrame.N
    best_trace = float('inf')
    activeKeyPointIndex = -1
    for i in range(N):
        cur_trace = np.trace(CurrentFrame.mPostMatrices_array[i])
        if cur_trace < best_trace:
            best_trace = cur_trace
            activeKeyPointIndex = i

    return activeKeyPointIndex

def selectFixedActiveKeyPoint(CurrentFrame, originKpIndex):
    N = CurrentFrame.N

    activeKeyPointIndex = -1
    for i in range(N):
        if CurrentFrame.originTrackedKpIndexs[i] == originKpIndex:
            activeKeyPointIndex = i
            break

    return activeKeyPointIndex

def targetPositionIncycolpean(input_withoutOffset):
    xL = input_withoutOffset[0]
    xR = input_withoutOffset[1]
    y = input_withoutOffset[2]

    return cam_baseline/(xL - xR) * np.array([0.5*(xL + xR), y, cam_focalLength])


#next best position for active point
def getNextBestPositionofFeature(input_withoutOffset, delta):

    targetPositionCy = targetPositionIncycolpean(input_withoutOffset)
    print("targetPosition in cycolpean frame: {0}".format(targetPositionCy))
    #
    # targetPositionS = R_s_c.dot(targetPositionCNext)
    #
    # targetPositionW = R_w_s.dot(targetPositionS) + t_w_s
    delta_norm = delta/np.linalg.norm(delta)

    print("delta_norm:  {0}".format(delta_norm))
    # delta_norm[0] = 0
    targetPositionCyNext = targetPositionCy - 0.25 * delta_norm  #0.25
    return targetPositionCy, targetPositionCyNext

#Compute camera's next position in world coordiante
def computeCameraTranslation(agent,targetCurrentPosition, targetNextPosition):
    twc1 = agent.state.sensor_states["left_sensor"].position  # twc at tk-1
    Rws1 = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    Rwc1 = Rws1.dot(R_s_c)
    Rwcy1 = Rwc1  # R(tk-1)

    r1 = twc1 - Rwcy1.dot(tcyclopean_leftcamera)  # r(tk-1) = twcy at tk-1

    pt = targetNextPosition - targetCurrentPosition
    t12_cy = -pt #1: tk-1 cy 2: tk cy

    delta_r = Rwcy1.dot(t12_cy)

    delta_norm = delta_r / np.linalg.norm(delta_r)
    # delta_r = 2*(-Rwcy1.dot(targetNextPosition - targetCurrentPosition))
    # delta_norm = delta_r / np.linalg.norm(delta_r)
    nextBestCameraPositionInCy = r1 + 0.25 * delta_norm

    print("delta_norm: {0}".format(delta_norm))
    # print("delta_tws: {0}, norm of delta_tws: {1}".format(delta_tws, np.linalg.norm(delta_tws)))
    print("nextBestCameraPostion: {0}".format(nextBestCameraPositionInCy))
    return nextBestCameraPositionInCy

#Having the z axis of the midpoint between the cameras point at the feature point
def rotateCameraToMidpoint(targetPositionCyNext_):
    if targetPositionCyNext_[0] < 0:
        tan_theta = abs(targetPositionCyNext_[0]/targetPositionCyNext_[2])
        theta = math.degrees(math.atan(tan_theta))
        theta_rad = np.deg2rad(theta)
        return np.array([[math.cos(theta_rad), 0, math.sin(theta_rad)], [0, 1, 0], [-math.sin(theta_rad), 0, math.cos(theta_rad)]])
    elif targetPositionCyNext_[0] > 0:
        tan_theta = abs(targetPositionCyNext_[0] / targetPositionCyNext_[2])
        theta = -1*math.degrees(math.atan(tan_theta))
        theta_rad = np.deg2rad(theta)
        return np.array(
            [[math.cos(theta_rad), 0, math.sin(theta_rad)], [0, 1, 0], [-math.sin(theta_rad), 0, math.cos(theta_rad)]])
    else:
        return np.identity(3)

def setAgentPosition(agent, nextBestCameraPostionIncy):

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R_w_c = R_w_s.dot(R_s_c)  # R_w_c

    nextBestCameraPosition = nextBestCameraPostionIncy + R_w_c.dot(tcyclopean_leftcamera)

    agent_state = agent.get_state()
    y = agent_state.position[1]
    # y = nextBestCameraPostionIncy[1]-1.5
    agent_state.position = np.array([nextBestCameraPosition[0], y, nextBestCameraPosition[2]])
    agent.set_state(agent_state)
    return agent.state

def setAgentRotation(agent, nextBestCameraRotation):
    R_w_s = nextBestCameraRotation.dot(R_s_c.T)

    agent_state = agent.get_state()
    agent_state.rotation = quaternion.from_rotation_matrix(R_w_s)
    # agent_state.sensor_states["left_sensor"].rotation = quaternion.from_rotation_matrix(R_w_s)
    # agent_state.sensor_states["right_sensor"].rotation = quaternion.from_rotation_matrix(R_w_s)
    # agent_state.sensor_states["left_sensor_depth"].rotation = quaternion.from_rotation_matrix(R_w_s)
    # agent_state.sensor_states["right_sensor_depth"].rotation = quaternion.from_rotation_matrix(R_w_s)

    agent.set_state(agent_state)
    return agent.state









def trackKeyPointIndex(CurrentFrame, preindex, depthIm):
    for i in range(len(CurrentFrame.preTrackedKpIndexs)):
        if CurrentFrame.preTrackedKpIndexs[i] == preindex:
            resIndex = i
            uL = CurrentFrame.Kps[i, 0]
            v = CurrentFrame.Kps[i, 1]
            depth_gt = depthIm[round(v), round(uL)]
            return resIndex, depth_gt
        # else:
        #     return -1, -1

    return -1, -1

def computeLocErrorForTrackedKeypoint(CurrentFrame, curTrackedKpIndex, Kp_loc_W_gt):
    uL = CurrentFrame.Kps[curTrackedKpIndex, 0]
    v = CurrentFrame.Kps[curTrackedKpIndex, 1]
    uR = CurrentFrame.KpRights[curTrackedKpIndex, 0]
    z_est = cam_baseline * cam_focalLength / (uL - uR)
    x_est = ((uL - cx) * z_est) / cam_focalLength
    y_est = ((v - cy) * z_est) / cam_focalLength
    loc_est = np.array([x_est, y_est, z_est])

    Rwc = CurrentFrame.getRotation()
    twc = CurrentFrame.getTranslation()

    loc_est_W = Rwc.dot(loc_est) + twc


    delta_position = Kp_loc_W_gt - loc_est_W
    t = np.linalg.norm(delta_position)
    return np.linalg.norm(delta_position)


def computeLocErrorForAllKeyPoints(CurrentFrame, depth_img):
    N = CurrentFrame.N
    loc_error_array = np.zeros(N)
    sum_error = 0
    notOccluded_num = 0
    for i in range(N):
        uL = CurrentFrame.Kps[i, 0]
        v = CurrentFrame.Kps[i, 1]
        uR = CurrentFrame.KpRights[i, 0]
        z_est = cam_baseline * cam_focalLength / (uL - uR)
        x_est = ((uL - cx) * z_est) / cam_focalLength
        y_est = ((v - cy) * z_est) / cam_focalLength
        loc_est = np.array([x_est, y_est, z_est])

        z_gt = depth_img[round(v), round(uL)]
        if z_gt <= 0.0:
            continue

        disparity = cam_baseline * cam_focalLength / z_gt
        v_gt = uL - disparity
        x_gt = ((uL - cx) * z_gt) / cam_focalLength
        y_gt = ((v_gt - cx) * z_gt) / cam_focalLength
        loc_gt = np.array([x_gt, y_gt, z_gt])

        error = np.linalg.norm((loc_gt - loc_est))
        sum_error += error
        notOccluded_num += 1
        loc_error_array[i] = error



    if notOccluded_num > 0:
        average_error = sum_error / notOccluded_num
        return average_error
    else:
        return -1

def targetWordCoordinate(CurrentFrame, uL, v, depth):
    zc = depth
    xc = zc * (uL - cx) / cam_focalLength  # xc 3d in cam
    yc = zc * (v - cy) / cam_focalLength  # yc

    targetPositionC = np.array([xc, yc, zc])
    Rwc = CurrentFrame.getRotation()
    twc = CurrentFrame.getTranslation()

    targetPositionW = Rwc.dot(targetPositionC) + twc

    # targetPositionS = R_s_c.dot(targetPositionC)

    # q = agent.state.sensor_states["left_sensor"].rotation
    # R_w_s = quaternion.as_rotation_matrix(q)
    #
    # t_w_s = agent.state.sensor_states["left_sensor"].position
    # # print("targetPosition sensor frame without translate value: {0}".format(R_w_s.dot(targetPositionS)))
    # targetPositionW = R_w_s.dot(targetPositionS) + t_w_s
    # print("targetPosition world value: {0}".format(targetPositionW))

    return targetPositionW

def project(CurrentFrame, mappointW):

    Rwc = CurrentFrame.getRotation()
    twc = CurrentFrame.getTranslation()
    Rcw = Rwc.T

    tcw = -Rcw.dot(twc)


    targetPositionC = Rcw.dot(mappointW)+tcw

    zc = targetPositionC[2]

    uL = (cam_focalLength * targetPositionC[0]) / zc + cx
    v = (cam_focalLength * targetPositionC[1]) / zc + cy

    uR = uL - ((cam_focalLength * cam_baseline) / zc)

    return np.array([uL, uR, v])
