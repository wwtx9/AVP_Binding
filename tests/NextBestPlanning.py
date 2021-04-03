import numpy as np
import quaternion

import habitat_sim
from numpy.linalg import inv
import math

import AVP_Binding as m

import matplotlib.pyplot as plt

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"

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

def isObserved(input):
    if input[0] < 0 or input[0] >= width or input[1] < 0 or input[1] >= width or input[2] < 0 or input[2] >= height:
        return False
    else:
        return True


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

    level = int(depth * 8 / 10)

    if level >= 8:
        level = 7

    Q = mvLevelSigma2[level] * np.identity(3)

    return Q

#Compute camera's next position in world coordiante
def computeCameraTranslation(agent,targetCurrentPosition, targetNextPosition):
    twc1 = agent.state.sensor_states["left_sensor"].position  # twc at tk-1
    Rws1 = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    Rwc1 = Rws1.dot(R_s_c)
    Rwcy1 = Rwc1  # R(tk-1)
    r1 = twc1 - Rwcy1.dot(tcyclopean_leftcamera)  # r(tk-1) = twcy at tk-1

    # r_star = r1 + Rwcy1.dot(targetNextPosition - targetCurrentPosition)
    pt = targetNextPosition - targetCurrentPosition
    t12_cy = -pt #1: tk-1 cy 2: tk cy

    delta_r = Rwcy1.dot(t12_cy)

    delta_norm = delta_r / np.linalg.norm(delta_r)
    # delta_r = 2*(-Rwcy1.dot(targetNextPosition - targetCurrentPosition))
    # delta_norm = delta_r / np.linalg.norm(delta_r)

    nextBestCameraPositionInCy = r1 + 0.25 * delta_norm

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


def computeLocError(agent, input, targetW_, Rwcy):
    input = np.array([input[0]-cx, input[1]-cx, input[2]-cy])
    targetCy = targetPositionIncycolpean(input)
    twc = agent.state.sensor_states["left_sensor"].position
    targetC = targetCy + tleftcamera_cyclopean
    targetPositionW = Rwcy.dot(targetC) + twc
    # print("targetPosition world value in this episode by taking fixed action: {0}".format(targetPositionW))
    delta_position = targetPositionW - targetW_
    mse_loc = np.linalg.norm(delta_position)
    return mse_loc

def plotTrace(y_data, id):
    n = len(y_data)
    x = range(1, n + 1)

    fig = plt.figure()
    plt.plot(x, y_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Steps")
    plt.ylabel("Trace")
    plt.yscale('log')
    plt.title("Experiment {0}'s Trace".format(id+1))
    plt.show()
    # fig.savefig('Experiment {0}.png'.format(id+1), dpi=fig.dpi)
    # fig.savefig('Experiment.png', dpi=fig.dpi)

def drawTrajectory(x_nbv, z_nbv, targetPositionW, exId):
    fig = plt.figure()

    plt.plot(x_nbv, z_nbv, color="y", linestyle="--", marker=".", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.plot(targetPositionW[0], targetPositionW[2], 'o', color='r')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend(["Gradient Descent", "Target"])
    # plt.yscale('log')
    plt.title("Experiment {0}'s trajectory".format(exId+1))
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    fig.savefig('Experiment_{0}_traj.png'.format(exId+1), dpi=fig.dpi)
    # fig.savefig('trajectory.png', dpi=fig.dpi)

def plotTraceTogether(y1_data, y2_data, id):
    n1 = len(y1_data)
    x1 = range(1, n1 + 1)

    n2 = len(y2_data)
    x2 = range(1, n2 + 1)
    fig = plt.figure()
    plt.plot(x1, y1_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    plt.plot(x2, y2_data, color="r", linestyle="--", marker="*", linewidth=1.0)

    plt.legend(["Gradient Descent", "Fixed move"], bbox_to_anchor=(0.2, 0.95))

    plt.xlabel("Steps")
    plt.ylabel("Trace")
    plt.yscale('log')
    plt.title("Experiment {0}'s Trace".format(id + 1))
    plt.show()
    fig.savefig('Experiment_{0}_trace.png'.format(id + 1), dpi=fig.dpi)

def plotLocError(y_data, id):
    n = len(y_data)
    x = range(1, n + 1)

    fig = plt.figure()
    plt.plot(x, y_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Steps")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.title("Exp {0}: Position Norm Error".format(id))
    plt.show()

def plotDistantFromTarget(y_data, id):
    n = len(y_data)
    x = range(1, n + 1)
    fig = plt.figure()
    plt.plot(x, y_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.legend(["Gradient Descent"], loc='best')
    plt.xlabel("Steps")
    plt.ylabel("Distance")

    plt.title("Exp {0}: Distance From Target".format(id+1))
    plt.show()
    # fig.savefig('Experiment_{0}_Distance.png'.format(id + 1), dpi=fig.dpi)


def plotLocErrorTogether(y1_data, y2_data, exId):
    n1 = len(y1_data)
    x1 = range(1, n1 + 1)

    n2 = len(y2_data)
    x2 = range(1, n2 + 1)
    fig = plt.figure()
    plt.plot(x1, y1_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    plt.plot(x2, y2_data, color="r", linestyle="--", marker="*", linewidth=1.0)

    plt.legend(["Gradient Descent", "Fixed move"], loc='best')

    plt.xlabel("Steps")
    plt.ylabel("Localization error")
    # plt.yscale('log')
    plt.title("Exp {0}: Localization error".format(exId+1))
    plt.show()
    fig.savefig('Experiment_{0}_locError.png'.format(exId + 1), dpi=fig.dpi)

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

    inputs = np.zeros((50, 3))

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

            if (index >= 50):
                break

    # print(sp.mCurrentFrame.mvKPs)
    #
    # print(len(sp.mCurrentFrame.mvuRight))
    init_state = agent.state
    print("init agent position: {0}".format(init_state.position))
    # start move
    for exId in range(0, 1):
        print("++++++++Experiment {0}+++++++".format(exId))
        agent.set_state(init_state)
        leftImg = leftImgOrg
        rightImg = rightImgOrg
        error_array = []
        trace_array = []
        dis_array = []

        x_nbv = []
        z_nbv = []

        # chose fix move and do gradient descent
        for i in range(0, 200):  # np.sqrt(loc_error) > 1e-3:
            print("+++++++++++++Step {0}++++++++".format(i))
            print("Current Agent's state: {0}".format(agent.state))
            obs = sim.get_sensor_observations()
            if i == 0:
                # input = np.array([inputs[exId][0], inputs[exId][1], inputs[exId][2]])
                input = np.array([198, 186, 373])
                targetPositionW = targetWordCoordinate(agent, 198, 186, 373)  # 917, 905, 322   198, 186, 373

                # targetPositionW = targetWordCoordinate(agent, input[0], input[1], input[2])  #left: 198, 186, 373 right: 917,905,322 437, 431, 293
            else:
                input = targetPixelInCurrentCamera(agent, targetPositionW)
                print("current input:{0}".format(input))
                if not isObserved(input):
                    print("Feature point can not be observed by camera")
                    break

            R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
            R_w_c = R_w_s.dot(R_s_c)

            err = computeLocError(agent, input, targetPositionW, R_w_c)
            error_array.append(err)

            dis = computeDistanceFromTarget(agent, targetPositionW)
            dis_array.append(dis)
            # print("loc error: {0}".format(err))

            #compute U_obs and posterior
            depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

            depth_pair = np.clip(depth_pair, 0, 10)

            U_obs = computeObsCovariance(agent, depth_pair, targetPositionW)

            if i == 0:
                U_prior = U_obs

            U_post, trace = objectiveFun(U_prior, U_obs)

            trace_array.append(trace)

            print("Current Agent's trace: {0}".format(trace))

            #visualize
            # obs = sim.get_sensor_observations()
            # stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
            # if len(stereo_pair.shape) > 2:
            #     stereo_pair = stereo_pair[..., 0:3][..., ::-1]
            # cv2.imshow("stereo_pair", stereo_pair)
            #
            # keystroke = cv2.waitKey(0)
            # if keystroke == ord("q"):
            #     break

            # Predict next best position of target and camera at time k+1 base on current observation at time k
            depth = depth_pair[int(input[2]), int(input[0])]
            print("depth: {0}".format(depth))
            input_withoutOffset = np.array([input[0] - cx, input[1] - cx, input[2] - cy])
            print("input_withoutOffset: {0}".format(input_withoutOffset))
            print("U_prior: {0}".format(U_prior))
            print("U_obs: {0}".format(U_obs))
            delta = sp.GradientForTarget(input_withoutOffset, depth, U_prior, R_w_c)
            print("delta: {0}".format(delta/np.linalg.norm(delta)))
            targetPositionCy, targetPositionCyNext = getNextBestPositionofFeature(input_withoutOffset, delta)
            print("targetPositionCNext in cyclopean coordinate: {0}".format(targetPositionCyNext))

            #compute next best position for camera
            nextBestCameraPostionIncy = computeCameraTranslation(agent, targetPositionCy, targetPositionCyNext)
            # compute next best rotation for camera
            R = rotateCameraToMidpoint(targetPositionCyNext) #Rck-1ck
            Rws = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
            Rwc = Rws.dot(R_s_c)
            Rwcy = Rwc  # R(tk-1)
            Rwcy_rectify = Rwcy.dot(R.T)

            x_nbv.append(nextBestCameraPostionIncy[0])
            z_nbv.append(nextBestCameraPostionIncy[2])

            # update camera position
            setAgentPosition(agent, nextBestCameraPostionIncy)

            setAgentRotation(agent, Rwcy_rectify)

            U_prior = U_post


        drawTrajectory(x_nbv, z_nbv, targetPositionW, exId)
        # plotLocError(error_array, exId)
        # plotTrace(trace_array, exId)

        # plotTraceTogether(trace_array, y2, exId)
        # plotDistantFromTarget(dis_array, exId)
        # plotLocErrorTogether(error_array, dis2, exId)

        print("Average error: {0}".format(np.mean(error_array)))

def computeDistanceFromTarget(agent, targetPositionW_):
    pos = agent.state.position
    # print("targetPosition world value in this episode by taking fixed action: {0}".format(targetPositionW))
    dis = targetPositionW_ - pos
    dis = np.linalg.norm(dis)
    return dis


def setAgentPosition(agent, nextBestCameraPostionIncy):

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R_w_c = R_w_s.dot(R_s_c)  # R_w_c

    nextBestCameraPosition = nextBestCameraPostionIncy + R_w_c.dot(tcyclopean_leftcamera)

    agent_state = agent.get_state()
    y = agent_state.position[1]
    # y = nextBestCameraPostionIncy[1]-1.5
    agent_state.position = np.array([nextBestCameraPosition[0]+cam_baseline/2, y, nextBestCameraPosition[2]])
    agent.set_state(agent_state)
    return agent.state  # agent.scene_node.transformation_matrix()

def setAgentRotation(agent, nextBestCameraRotation):
    R_w_s = nextBestCameraRotation.dot(R_s_c.T)

    agent_state = agent.get_state()
    agent_state.rotation = quaternion.from_rotation_matrix(R_w_s)
    agent_state.sensor_states["left_sensor"].rotation = quaternion.from_rotation_matrix(R_w_s)
    agent_state.sensor_states["right_sensor"].rotation = quaternion.from_rotation_matrix(R_w_s)
    agent_state.sensor_states["left_sensor_depth"].rotation = quaternion.from_rotation_matrix(R_w_s)
    agent_state.sensor_states["right_sensor_depth"].rotation = quaternion.from_rotation_matrix(R_w_s)

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


def targetWordCoordinate(agent, uL, uR, v):
    # targetPosition in left camera coordinate
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


    xR = xL - ((cam_focalLength * cam_baseline) / zc)

    t = cam_focalLength*cam_baseline/(xL - xR)
    return np.array([xL, xR, y])


# def project(agent, targetPositionW):
#     R_c_s = R_s_c.T
#     R_w_sr = quaternion.as_rotation_matrix(agent.state.sensor_states["right_sensor"].rotation)
#     t_w_sr = agent.state.sensor_states["right_sensor"].position
#     R_sr_w = R_w_sr.T
#     t_sr_w = -1 * R_sr_w.dot(t_w_sr)
#     targetPositionS = R_sr_w.dot(targetPositionW) + t_sr_w  # sensor
#     targetPositionC = R_c_s.dot(targetPositionS)
#     xR = (cam_focalLength * targetPositionC[0]) / targetPositionC[2] + cx
#     y = (cam_focalLength * targetPositionC[1]) / targetPositionC[2] + cy
#     return np.array([xR, y])

def computeObsCovariance(agent, depth_pair, targetPositionW):

    pixels = targetPixelInCurrentCamera(agent, targetPositionW)

    pixels_withoutOffset = np.array([pixels[0] - cx, pixels[1] - cx, pixels[2] - cy])

    J = makeJacobian(pixels_withoutOffset)
    Q = makeQ(depth_pair[int(pixels[2]), int(pixels[0])])

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)

    R = R_w_s.dot(R_s_c)

    U_obs = np.linalg.multi_dot([R, J, Q, J.T, R.T])

    return U_obs

# update posterior and computer objective function
def objectiveFun(U_prior, U_obs):
    U_post = (inv(U_prior) + inv(U_obs))
    U_post = inv(U_post)

    return U_post, U_post[0, 0] + U_post[1, 1] + U_post[2, 2]


def targetPositionIncycolpean(input_withoutOffset):
    xL = input_withoutOffset[0]
    xR = input_withoutOffset[1]
    y = input_withoutOffset[2]

    return cam_baseline/(xL - xR) * np.array([0.5*(xL + xR), y, cam_focalLength])

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()
    setupAgentwithSensors(display=args.display)
