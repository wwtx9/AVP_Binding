import numpy as np
import quaternion

import habitat_sim
from numpy.linalg import inv
import math

import AVP_Binding as m
import cv2
import matplotlib.pyplot as plt

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"

cam_baseline = 0.2
cam_focalLength = 450
height = 640
width = 960

dic = {0: "move_forward", 1: "turn_left", 2: "turn_right"}

R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])


def isObserved(input):
    if input[0] < 0 or input[0] >= width or input[1] < 0 or input[1] >= width or input[2] < 0 or input[2] >= height:
        return False
    else:
        return True


def makeJacobian(input, camBaseline):

    xL = input[0]
    xR = input[1]
    y = input[2]

    cx = width / 2
    cy = height / 2

    front_J = cam_baseline / ((xL - xR) ** 2)

    J = np.array([[cx - xR, xL - cx, 0],
                  [cy - y, y - cy, xL - xR],
                  [-camBaseline, camBaseline, 0]])

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

    level = int(depth * 8 / 10)

    if level >= 8:
        level = 7

    Q = mvLevelSigma2[level] * np.identity(3)

    return Q


# Helper function to render observations from the stereo agent
def _render(sim, isdepth=False):
    agent = sim.get_agent(0)

    obs = sim.get_sensor_observations()

    sy = m.System(
        "/home/wangweihan/PycharmProjects/AVP_Python/habitat.yaml")

    leftImgOrg = obs["left_sensor"]
    if len(leftImgOrg.shape) > 2:
        # change image from rgb to bgr
        leftImgOrg = leftImgOrg[..., 0:3][..., ::-1]

    rightImgOrg = obs["right_sensor"]
    if len(rightImgOrg.shape) > 2:
        # change image from rgb to bgr
        rightImgOrg = rightImgOrg[..., 0:3][..., ::-1]

    # Extract feature point
    sy.ProcessingStereo(leftImgOrg, rightImgOrg, 0.1)
    sp = sy.mpSparsePipline
    print(len(sp.mCurrentFrame.mvKPs))

    inputs = np.zeros((10, 3))

    index = 0
    for i in range(len(sp.mCurrentFrame.mvKPs)):
        # sp.mCurrentFrame.mvuRight[i] > -1 means this keypoint in left image has a corresponding keypoint in right
        if sp.mCurrentFrame.mvuRight[i] > -1:
            xL = round(sp.mCurrentFrame.mvKPs[i][0])
            xR = round(sp.mCurrentFrame.mvuRight[i])
            y = round(sp.mCurrentFrame.mvKPs[i][1])

            inputs[index][0] = xL
            inputs[index][1] = xR
            inputs[index][2] = y
            index = index + 1

            if (index >= 10):
                break

    # print(sp.mCurrentFrame.mvKPs)
    #
    # print(len(sp.mCurrentFrame.mvuRight))
    init_state = agent.state

    # start move
    for exId in range(0, 1):
        print("++++++++Experiment {0}+++++++".format(exId))
        agent.set_state(init_state)
        leftImg = leftImgOrg
        rightImg = rightImgOrg

        stereo_pair = np.concatenate([leftImg, rightImg], axis=1)

        cv2.imshow("stereo_pair", stereo_pair)
        cv2.waitKey(1000)

        # time k = 0
        x_nbv = []
        z_nbv = []
        agent.set_state(init_state)

        # new feature point in pixel coordinate
        # input = np.array([int(inputs[exId][0]), int(inputs[exId][1]), int(inputs[exId][2])])
        targetPositionW = targetWordCoordinate(agent, 917, 905, 322) #917, 905, 322   198, 186, 373
        # targetPositionW = targetWordCoordinate(agent, input[0], input[1], input[2])  #left: 198, 186, 373 right: 917,905,322 437, 431, 293

        print("project on right image pixel: {0}".format(project(agent, targetPositionW)))

        depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

        depth_pair = np.clip(depth_pair, 0, 10)

        # test
        input = np.array([917, 905, 322])

        # depth of target
        depth = depth_pair[input[2], input[0]]

        J = makeJacobian(input, cam_baseline)

        Q = makeQ(depth)

        U_obs = computeCovarianceByPixel(agent, J, Q)

        x_nbv.append(agent.state.position[0])
        z_nbv.append(agent.state.position[2])


        # Next target and camera position, timestep k = 1
        U_prior = U_obs

        R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
        R_w_c = R_w_s.dot(R_s_c)  # R_w_c

        # Uk+1|k, Uobs
        delta = sp.Gradient(input, depth, U_prior, R_w_c)

        #predict next best target position and camera position
        targetPositionCNext = getNextBestPositionofFeature(agent, targetPositionW, delta)
        print("targetPositionCNext: {0}".format(targetPositionCNext))

        nextBestCameraPostion = computeCameraPostion(agent, targetPositionCNext, targetPositionW)

        x_nbv.append(nextBestCameraPostion[0])
        z_nbv.append(nextBestCameraPostion[2])

        # update camera position
        setAgentPosition(agent, nextBestCameraPostion)
        # obs = sim.step("turn_left")
        # obs = sim.step("turn_left")
        obs = sim.step("turn_right")
        obs = sim.step("turn_right")
        obs = sim.step("turn_right")

        print("Current Agent's left camera state after Gradient Descend without rotation: {0}".format(agent.state.sensor_states["left_sensor"]))

        # chose fix move and do gradient descent
        for i in range(1, 40):  # np.sqrt(loc_error) > 1e-3:
            print("+++++++++++++{0}++++++++".format(i))

            obs = sim.get_sensor_observations()

            input = targetPixelInCurrentCamera(agent, targetPositionW)

            if not isObserved(input):
                print("Feature point can not be observed by camera")
                break
            else:
                depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

                depth_pair = np.clip(depth_pair, 0, 10)

                U_obs = computeObsCovariance(agent, depth_pair, targetPositionW)

                # visualize
                stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
                if len(stereo_pair.shape) > 2:
                    stereo_pair = stereo_pair[..., 0:3][..., ::-1]
                cv2.imshow("stereo_pair", stereo_pair)

                cv2.waitKey(1000)

                R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
                R_w_c = R_w_s.dot(R_s_c)  # R_w_c

                # Predict next best position of target and camera at time k+1 base on current observation at time k
                depth = depth_pair[input[2], input[0]]
                delta = sp.Gradient(input, depth, U_prior, R_w_c)
                targetPositionCNext = getNextBestPositionofFeature(agent, targetPositionW, delta)
                print("targetPositionCNext: {0}".format(targetPositionCNext))

                nextBestCameraPostion = computeCameraPostion(agent, targetPositionCNext, targetPositionW)
                x_nbv.append(nextBestCameraPostion[0])
                z_nbv.append(nextBestCameraPostion[2])

                # update camera position
                setAgentPosition(agent, nextBestCameraPostion)
                print("Current Agent's left camera state after Gradient Descend without rotation: {0}".format(agent.state.sensor_states["left_sensor"]))

                U_post, trace = objectiveFun(U_prior, U_obs)

                U_prior = U_post

            # visualize
            # obs = sim.get_sensor_observations()
            #
            # stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
            # if len(stereo_pair.shape) > 2:
            #     stereo_pair = stereo_pair[..., 0:3][..., ::-1]
            #
            # cv2.imshow("stereo_pair", stereo_pair)
            # keystroke = cv2.waitKey(1000)

        # plotResult(y_data, exId)
        drawTrajectory(x_nbv, z_nbv, targetPositionW, exId)


def computeCameraPostion(agent, targetPositionNext_, targetPositionW_):

    R_w_s = quaternion.as_rotation_matrix(agent.state.sensor_states["left_sensor"].rotation)

    tws = agent.state.sensor_states["left_sensor"].position

    targetPositionNextS_ = R_s_c.dot(targetPositionNext_)

    delta_tws = 2 * (R_w_s.dot((targetPositionNextS_)) + tws - targetPositionW_)

    delta_tws = delta_tws / np.linalg.norm(delta_tws)

    nextBestCameraPosition = tws - 0.25 * delta_tws

    # print("delta_tws: {0}, norm of delta_tws: {1}".format(delta_tws, np.linalg.norm(delta_tws)))
    print("nextBestCameraPostion: {0}".format(nextBestCameraPosition))
    return nextBestCameraPosition

def setAgentPosition(agent, NBCameraPosition):
    agent_state = agent.get_state()
    agent_state.position = np.array([NBCameraPosition[0]+cam_baseline/2, NBCameraPosition[1]-1.5, NBCameraPosition[2]])
    agent.set_state(agent_state)
    return agent.state  # agent.scene_node.transformation_matrix()


# visualize the target
def drawTarget(input, leftIm, rightIm):
    r = 5
    leftIm = np.array(leftIm)
    rightIm = np.array(rightIm)

    xL = round(input[0])
    y = round(input[2])

    x1 = xL - r
    y1 = y - r

    x2 = xL + r
    y2 = y + r

    cv2.rectangle(leftIm, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.circle(leftIm, (xL, y), 2, (0, 0, 255), -1)

    xR = round(input[1])

    xR1 = xR - r
    yR1 = y - r

    xR2 = xR + r
    yR2 = y + r

    cv2.rectangle(rightIm, (xR1, yR1), (xR2, yR2), (0, 0, 255))
    cv2.circle(rightIm, (xR, y), 2, (0, 0, 255), -1)

    return leftIm, rightIm


# visualize feature points
def drawFeaturePoints(input, leftIm, rightIm, KPs, uRights):
    r = 5
    leftIm = np.array(leftIm)
    rightIm = np.array(rightIm)

    for i in range(len(KPs)):

        if uRights[i] > -1:
            xL = round(KPs[i][0])
            y = round(KPs[i][1])

            x1 = xL - r
            y1 = y - r

            x2 = xL + r
            y2 = y + r

            if (xL == input[0] and y == input[2]):  # target red
                cv2.rectangle(leftIm, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.circle(leftIm, (xL, y), 2, (0, 0, 255), -1)
            else:
                cv2.rectangle(leftIm, (x1, y1), (x2, y2), (0, 255, 0))
                cv2.circle(leftIm, (xL, y), 2, (0, 255, 0), -1)

            xR = round(uRights[i])

            xR1 = xR - r
            yR1 = y - r

            xR2 = xR + r
            yR2 = y + r
            if (xR == input[1] and y == input[2]):
                cv2.rectangle(rightIm, (xR1, yR1), (xR2, yR2), (0, 0, 255))
                cv2.circle(rightIm, (xR, y), 2, (0, 0, 255), -1)
            else:
                cv2.rectangle(rightIm, (xR1, yR1), (xR2, yR2), (0, 255, 0))
                cv2.circle(rightIm, (xR, y), 2, (0, 255, 0), -1)

    return leftIm, rightIm


# Rotate camera to midpoint of the baseline goes through the feature point
def rotateCameraToMidpoint(agent, sim, targetPositionW_):
    pixel = targetPixelInCurrentCamera(agent, targetPositionW_)

    uL = pixel[0]
    uR = pixel[1]
    v = pixel[2]

    if uL == -1 and uR == -1 and v == -1:
        return False, sim.get_sensor_observations()

    cx = width / 2

    zc = cam_focalLength * cam_baseline / (uL - uR)
    xc = zc * (uL - cx) / cam_focalLength  # xc 3d in cam

    xc_now = xc
    cur_error = abs(xc_now - cam_baseline / 2)
    pre_error = float('inf')
    isLeft = True
    nochange = False
    while pre_error >= cur_error:
        if xc_now > cam_baseline / 2:
            sim.step("turn_right")
            isLeft = True
        elif xc_now < cam_baseline / 2:
            sim.step("turn_left")
            isLeft = False
        else:
            nochange = True
            break
        pre_error = cur_error

        pixel_rec = targetPixelInCurrentCamera(agent, targetPositionW_)

        uL = pixel_rec[0]
        uR = pixel_rec[1]

        cx = width / 2

        zc = cam_focalLength * cam_baseline / (uL - uR)
        xc_rec = zc * (uL - cx) / cam_focalLength  # xc 3d in cam
        cur_error = abs(xc_rec - cam_baseline / 2)

    if isLeft and not nochange:
        print("Rotate camera to right so that z axis of midpoint of baseline go through feature point")
        return True, sim.step("turn_left")
    elif not isLeft and not nochange:
        print("Rotate camera to left so that z axis of midpoint of baseline go through feature point")
        return True, sim.step("turn_right")


def computeTargetPositionOnWorldFrame(agent, targetPositionW_):

    pixels = targetPixelInCurrentCamera(agent, targetPositionW_)

    if isObserved(pixels):
        cx = width / 2
        cy = height / 2

        uL = pixels[0]
        uR = pixels[1]
        v = pixels[2]

        zc = cam_focalLength * cam_baseline / (uL - uR)
        xc = zc * (uL - cx) / cam_focalLength  # xc 3d in cam
        yc = zc * (v - cy) / cam_focalLength  # yc

        targetPositionC = np.array([xc, yc, zc])

        # print("targetPosition camera frame: {0}".format(targetPositionC))
        targetPositionS = R_s_c.dot(targetPositionC)

        q = agent.state.sensor_states["left_sensor"].rotation
        R_w_s = quaternion.as_rotation_matrix(q)

        t_w_s = agent.state.sensor_states["left_sensor"].position
        # print("targetPosition sensor frame without translate value: {0}".format(R_w_s.dot(targetPositionS)))
        targetPositionW = R_w_s.dot(targetPositionS) + t_w_s
        # print("targetPosition world value in this episode by taking fixed action: {0}".format(targetPositionW))
        delta_position = targetPositionW - targetPositionW_
        mse_loc = np.linalg.norm(delta_position)
        # print("Mean Square Error: {0}".format(mse_loc))
        return True, targetPositionW, mse_loc
    else:
        return False, -1, -1


def setupAgentwithSensors(display=True):
    global cv2
    # Only import cv2 if we are doing to display
    if display:
        import cv2 as _cv2

        cv2 = _cv2

        cv2.namedWindow("stereo_pair")
        # cv2.namedWindow("depth_pair")
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene.id = (
        "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb")  # data/scene_datasets/habitat-test-scenes/skokloster-castle.glb

    # cam_baseline = 0.2
    # cam_focalLength = 450
    # height = 640
    # width = 960
    # constructPyramid()

    hfov = 2 * math.degrees(math.atan(width / (2 * cam_focalLength)))
    vfov = 2 * math.degrees(math.atan(height / (2 * cam_focalLength)))
    # First, let's create a stereo RGB agent
    left_rgb_sensor = habitat_sim.SensorSpec()
    # Give it the uuid of left_sensor, this will also be how we
    # index the observations to retrieve the rendering from this sensor
    left_rgb_sensor.uuid = "left_sensor"
    left_rgb_sensor.resolution = [height, width]
    left_rgb_sensor.parameters["hfov"] = str(hfov)
    left_rgb_sensor.parameters["vfov"] = str(vfov)
    # The left RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the left of the center of the agent
    left_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + (cam_baseline / 2) * habitat_sim.geo.LEFT

    # Same deal with the right sensor
    right_rgb_sensor = habitat_sim.SensorSpec()
    right_rgb_sensor.uuid = "right_sensor"
    right_rgb_sensor.resolution = [height, width]
    right_rgb_sensor.parameters["hfov"] = str(hfov)
    right_rgb_sensor.parameters["vfov"] = str(vfov)
    # The right RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the right of the center of the agent
    right_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + (cam_baseline / 2) * habitat_sim.geo.RIGHT

    # Now let's do the exact same thing but for a depth camera stereo pair!
    left_depth_sensor = habitat_sim.SensorSpec()
    left_depth_sensor.uuid = "left_sensor_depth"
    left_depth_sensor.resolution = [height, width]
    left_depth_sensor.parameters["hfov"] = str(hfov)
    left_depth_sensor.parameters["vfov"] = str(vfov)
    left_depth_sensor.position = 1.5 * habitat_sim.geo.UP + (cam_baseline / 2) * habitat_sim.geo.LEFT
    # The only difference is that we set the sensor type to DEPTH
    left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    right_depth_sensor = habitat_sim.SensorSpec()
    right_depth_sensor.uuid = "right_sensor_depth"
    right_depth_sensor.resolution = [height, width]
    right_depth_sensor.parameters["hfov"] = str(hfov)
    right_depth_sensor.parameters["vfov"] = str(vfov)
    right_depth_sensor.position = 1.5 * habitat_sim.geo.UP + (cam_baseline / 2) * habitat_sim.geo.RIGHT
    # The only difference is that we set the sensor type to DEPTH
    right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    agent_config = habitat_sim.AgentConfiguration()  # set configuration for agent (id = 0)
    agent_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor, left_depth_sensor, right_depth_sensor]

    sim = habitat_sim.simulator.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

    # set agent position
    inital_state_agent = place_agent(sim, agent_config)
    # print("inital agent state: {0} ".format(inital_state_agent))
    print("inital agent's sensor state: {0} ".format(inital_state_agent.sensor_states))
    # set action
    _render(sim, isdepth=True)


def place_agent(sim, agent_config):
    # place our agent in the scene
    agent_state = habitat_sim.agent.AgentState()

    # agent_state.position = [-1.7926959 ,  0.11083889, 19.255245 ] inital

    agent_state.position = np.array([-1.7926959, 0.11083889, 19.255245])  # [-4.1336217 ,  0.2099186, 12.3464   ] now
    # agent_state.sensor_states["left_sensor"].rotation = np.quaternion(0, 0, 1, 0)
    # agent_state.rotation = np.quaternion(1, 0, 2.45858027483337e-05, 0) 0.0871558338403702, 0, -0.996194660663605, 0

    agent_state.rotation = np.quaternion(1, 0, 0.0, 0)
    agent = sim.initialize_agent(0, agent_state)
    agent.get_state()
    agent.agent_config = agent_config
    return agent.state  # agent.scene_node.transformation_matrix()


def setAgentPosition(agent, NBCameraPosition):
    agent_state = agent.get_state()
    agent_state.position = np.array(
        [NBCameraPosition[0] + cam_baseline / 2, NBCameraPosition[1] - 1.5, NBCameraPosition[2]])
    agent.set_state(agent_state)
    return agent.state  # agent.scene_node.transformation_matrix()


def setAgentPositionByState(agent, last_agent_state):
    agent.set_state(last_agent_state)
    return agent.state  # agent.scene_node.transformation_matrix()


def targetWordCoordinate(agent, uL, uR, v):
    # targetPosition in left camera coordinate
    cx = width / 2
    cy = height / 2
    zc = cam_focalLength * cam_baseline / (uL - uR)
    xc = zc * (uL - cx) / cam_focalLength  # xc 3d in cam
    yc = zc * (v - cy) / cam_focalLength  # yc

    targetPositionC = np.array([xc, yc, zc])

    print("targetPosition camera frame: {0}".format(targetPositionC))

    targetPositionS = R_s_c.dot(targetPositionC)

    q = agent.state.sensor_states["left_sensor"].rotation
    R_w_s = quaternion.as_rotation_matrix(q)

    t_w_s = agent.state.sensor_states["left_sensor"].position
    # print("targetPosition sensor frame without translate value: {0}".format(R_w_s.dot(targetPositionS)))
    targetPositionW = R_w_s.dot(targetPositionS) + t_w_s
    print("targetPosition world value: {0}".format(targetPositionW))

    return targetPositionW


def targetPixelInCurrentCamera(agent, targetPositionW):
    cx = width / 2
    cy = height / 2

    R_c_s = R_s_c.T

    R_w_s = quaternion.as_rotation_matrix(agent.state.sensor_states["left_sensor"].rotation)

    R_s_w = R_w_s.T
    t_w_s = agent.state.sensor_states["left_sensor"].position
    t_s_w = -1 * R_s_w.dot(t_w_s)

    targetPositionS = R_s_w.dot(targetPositionW) + t_s_w

    targetPositionC = R_c_s.dot(targetPositionS)

    zc = targetPositionC[2]

    xL = (cam_focalLength * targetPositionC[0]) / zc + cx
    y = (cam_focalLength * targetPositionC[1]) / zc + cy
    xL = round(xL)
    y = round(y)

    xR = xL - ((cam_focalLength * cam_baseline) / zc)
    xR = round(xR)

    return np.array([int(xL), int(xR), int(y)])


def project(agent, targetPositionW):
    cx = width / 2
    cy = height / 2

    R_c_s = R_s_c.T
    R_w_sr = quaternion.as_rotation_matrix(agent.state.sensor_states["right_sensor"].rotation)
    t_w_sr = agent.state.sensor_states["right_sensor"].position
    R_sr_w = R_w_sr.T
    t_sr_w = -1 * R_sr_w.dot(t_w_sr)
    targetPositionS = R_sr_w.dot(targetPositionW) + t_sr_w  # sensor
    targetPositionC = R_c_s.dot(targetPositionS)
    xR = (cam_focalLength * targetPositionC[0]) / targetPositionC[2] + cx
    y = (cam_focalLength * targetPositionC[1]) / targetPositionC[2] + cy
    return np.array([xR, y])


def computeObsCovariance(agent, depth_pair, targetPositionW):

    pixels = targetPixelInCurrentCamera(agent, targetPositionW)

    J = makeJacobian(pixels, cam_baseline)

    Q = makeQ(depth_pair[pixels[2], pixels[0]])

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)

    R = R_w_s.dot(R_s_c)

    U_obs = np.linalg.multi_dot([R, J, Q, J.T, R.T])

    return U_obs


# init
def computeCovarianceByPixel(agent, J, Q):
    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R = R_w_s.dot(R_s_c)

    U_obs = np.linalg.multi_dot([R, J, Q, J.T, R.T])

    return U_obs


# update posterior and computer objective function
def objectiveFun(U_prior, U_obs):
    U_post = (inv(U_prior) + inv(U_obs))
    U_post = inv(U_post)

    return U_post, U_post[0, 0] + U_post[1, 1] + U_post[2, 2]


def objectiveFunInit(U_obs):
    return U_obs[0, 0] + U_obs[1, 1] + U_obs[2, 2]


def getNextBestPositionofFeature(agent, targetPositionW, delta):
    cx = width / 2
    cy = height / 2

    # R_c_s = R_s_c.T
    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R_w_c = R_w_s.dot(R_s_c)  # R_w_c

    R_c_s = R_s_c.T

    R_s_w = R_w_s.T
    t_w_s = agent.state.sensor_states["left_sensor"].position
    t_s_w = -1 * R_s_w.dot(t_w_s)

    targetPositionS = R_s_w.dot(targetPositionW) + t_s_w

    targetPositionC = R_c_s.dot(targetPositionS)

    # sum_delta = np.zeros(3)

    # while(np.linalg.norm(sum_delta) < 0.25):
    #     sum_delta += delta * 0.25
    sum_delta = delta * 0.25 / np.linalg.norm(delta)
    # print("norm sum_delta: {0}".format(np.linalg.norm(sum_delta)))
    #
    # print("norm length: {0}".format(np.linalg.norm(t_w_s)))
    targetPositionCNext = targetPositionC - sum_delta

    targetPositionS = R_s_c.dot(targetPositionCNext)

    targetPositionW = R_w_s.dot(targetPositionS) + t_w_s

    return targetPositionCNext


def nextBestCameraPosition(agent, delta):
    # need Rwc twc=tws
    R_c_s = R_s_c.T

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R_w_c = R_w_s.dot(R_s_c)  # R_w_c
    R_s_w = R_w_s.T
    t_w_s = agent.state.sensor_states["left_sensor"].position

    t_s_w = -R_s_w.dot(t_w_s)

    t_c_w = R_c_s.dot(t_s_w)

    sum_delta = 0.25 * (delta / np.linalg.norm(delta))
    print("sum_delta {0}".format(sum_delta))
    NBCameraPosition = -R_w_c.dot(t_c_w - sum_delta)

    return NBCameraPosition


def getGradientRespectToTranslation(agent, targetPositionW, NBV):
    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)

    t_w_s = agent.state.sensor_states["left_sensor"].position

    R = R_w_s.dot(R_s_c)

    sum_delta = np.zeros(3)

    delta = 2 * (R.dot(NBV) - targetPositionW + t_w_s)

    while (np.linalg.norm(sum_delta) < 0.25):
        sum_delta += delta

    return sum_delta


# plot the result
def plotResult(y_data, id):
    n = len(y_data)
    x = range(1, n + 1)

    fig = plt.figure()
    plt.plot(x, y_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Episode")
    plt.ylabel("Error")
    # plt.yscale('log')
    plt.title("Exp {0}: Position Norm Error".format(id))
    plt.show()


def drawTrajectory(x_nbv, z_nbv, targetPositionW, exId):
    fig = plt.figure()

    plt.plot(x_nbv, z_nbv, color="y", linestyle="--", marker=".", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.plot(targetPositionW[0], targetPositionW[2], 'o', color='r')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend(["fix action", "NBV"])
    plt.yscale('log')
    plt.title("trajectory {0}".format(exId))
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()
    setupAgentwithSensors(display=args.display)
