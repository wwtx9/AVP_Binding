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

mvInvLevelSigma2 = np.ones(8)
mvLevelSigma2 = np.ones(8)
mvScaleFactor = np.ones(8)
mvInvScaleFactor = np.ones(8)

def constructPyramid():
    for i in range(1,8):
        mvScaleFactor[i] = mvScaleFactor[i - 1] * 1.2
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i]

    for i in range(8):
        mvInvScaleFactor[i] = 1.0 / mvScaleFactor[i]
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i]


# Helper function to render observations from the stereo agent
def _render(sim, isdepth=False):

    agent = sim.get_agent(0)

    obs = sim.get_sensor_observations()

    sy = m.System(
        "/home/wangweihan/PycharmProjects/AVP_Python/habitat.yaml")


    leftImg = obs["left_sensor"]
    if len(leftImg.shape) > 2:
        # change image from rgb to bgr
        leftImg = leftImg[..., 0:3][..., ::-1]

    rightImg = obs["right_sensor"]
    if len(rightImg.shape) > 2:
        # change image from rgb to bgr
        rightImg = rightImg[..., 0:3][..., ::-1]

    # Extract feature point
    sy.ProcessingStereo(leftImg, rightImg, 0.1)
    sp = sy.mpSparsePipline
    print(len(sp.mCurrentFrame.mvKPs))

    inputs = np.zeros((10,3))

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

             if(index>=10):
                break

    # print(sp.mCurrentFrame.mvKPs)
    #
    # print(len(sp.mCurrentFrame.mvuRight))
    init_state = agent.state

    for x in range(0, 10):
        print("++++++++Experiment {0}+++++++".format(x))
        agent.set_state(init_state)
        xL = int(inputs[x][0])
        xR = int(inputs[x][1])
        y = int(inputs[x][2])
        print("Target pixel xL: {0}, xR: {1}, y:{2}".format(xL, xR, y))
        targetPositionW = targetWordCoordinate(agent, xL, xR, y )  #left: 198, 186, 373 right: 917,905,322 437, 431, 293

        print("project on right image pixel: {0}".format(project(agent, targetPositionW)))


        print("Current agent's sensor state in {0} Episode: {1} ".format(0, agent.state.sensor_states))
        print("Current agent's  state in {0} Episode: {1} ".format(0, agent.state))


        depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)


        depth_pair = np.clip(depth_pair, 0, 10)

        input = np.array([xL, xR, y])
        isValid, U_obs = computeCovarianceByPixel(agent, depth_pair, input)
        traceInit = objectiveFunInit(U_obs)

        R_s_c = np.array([[1, 0, 0], [0, -1, 0],
                          [0, 0, - 1]])  # c is not left camera here. it is coordinate conresponding to paper but it is same

        R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
        R_w_c = R_w_s.dot(R_s_c)  # R_w_c

        #gradient of feature point position
        # delta = sp.Gradient(input, depth, U_obs, R_w_c)
        # UobsNBV, targetPositionCNext = testGradient(agent, targetPositionW, delta)
        # Upost_NBV, traceNBV = objectiveFun(U_obs, UobsNBV)
        # print("traceNBV: {0}".format(traceNBV))
        # print("In first episode, the trace of convariance matrix of observation  is: {0} ".format(traceInit))
        # nbv_loc = agent.state.sensor_states["left_sensor"].position - getGradientRespectToTranslation(agent,targetPositionW, targetPositionCNext)
        # print("nbv camera poistion in world coordinate: {0}".format(nbv_loc))
        last_agent_state = agent.state
        # print("Current Agent's left camera state: {0}".format(agent.state.sensor_states["left_sensor"]))
        # currentCameraPosition = agent.state.sensor_states["left_sensor"].position
        # print("currentCameraPosition: {0}".format(currentCameraPosition))
        # #Gradient of translation of camera
        # depth = depth_pair[373, 198]
        # delta_camera = sp.GradientofCameraTranslation(input, depth, U_obs, R_w_c)
        # # print("input: {0}, depth: {1}, U:{2}, Rwc: {3}".format(input, depth, U_obs, R_w_c))
        # # print("delta_camera: {0}".format(delta_camera))
        # # print("delta_camera: {0}".format(delta_camera))
        # NBCameraPosition = nextBestCameraPosition(agent, delta_camera)
        # # print("NBCameraPosition: {0}".format(NBCameraPosition))
        # setAgentPosition(agent, NBCameraPosition)
        # print("Current Agent's left camera state after Gradient Descend: {0}".format(agent.state.sensor_states["left_sensor"]))
        # #isTrack, _ = rotateCameraToMidpoint(agent, sim, targetPositionW)
        # isTrack, _, loc_errorNB = computeTargetPositionOnWorldFrame(agent, targetPositionW)
        # #print("Current Agent's left camera state after Gradient Descend and adjustment: {0}".format(agent.state.sensor_states["left_sensor"]))
        # # print("Current Agent's left camera state after Gradient Descend: {0}".format(agent.state.sensor_states["left_sensor"]))
        #
        # #back to previous agent state
        # setAgentPositionByState(agent, last_agent_state)
        # print("Current Agent's left camera state with reback: {0}".format(agent.state.sensor_states["left_sensor"]))


        U_prior = U_obs


        input = np.array([xL, xR, y])
        leftIm, rightIm = drawFeaturePoints(input, leftImg, rightImg, sp.mCurrentFrame.mvKPs, sp.mCurrentFrame.mvuRight)
        stereo_pair = np.concatenate([leftIm, rightIm], axis=1)


        # cv2.imshow("stereo_pair", stereo_pair)
        # keystroke = cv2.waitKey(1000)
        # if keystroke == ord("q"):
        #     break

        # for i in range(1, 40): traceBest > 1e-3:
        i = 1
        y_data = np.array([])
        traceBest = float('inf')
        loc_error = float('inf')
        while loc_error > 1e-3 and i <= 50:
            print("Episode {0}".format(i))

            # print("Mean Square Error NB: {0}".format(loc_error_NB))

            bestIndex = -1
            U_post = np.identity(3)  # init U_obs
            # traceBest = float('inf')

            #select a fix move with best value of objective function
            for j in range(3):
                if j == 0:
                    obs = sim.step("move_forward")
                    # print("Try to move forward")

                    depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

                    if isdepth:
                        depth_pair = np.clip(depth_pair, 0, 10)

                    isValidF, U_obsF = computeCovariance(agent, depth_pair, targetPositionW)

                    U_postF, traceF = objectiveFun(U_prior, U_obsF)


                    if traceF < traceBest and isValidF:
                        traceBest = traceF
                        bestIndex = j
                        U_post = U_postF

                    agent.set_state(last_agent_state)
                elif j == 1:
                    obs = sim.step("turn_left")
                    # print("Try to Left")

                    depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

                    # If it is a depth pair, manually normalize into [0, 1]
                    # so that images are always consistent
                    if isdepth:
                        depth_pair = np.clip(depth_pair, 0, 10)
                        # depth_pair /= 10.0

                    isValidL, U_obsL = computeCovariance(agent, depth_pair, targetPositionW)

                    U_postL, traceL = objectiveFun(U_prior, U_obsL)

                    if traceL < traceBest and isValidL:
                        traceBest = traceL
                        bestIndex = j
                        U_post = U_postL
                        #print("Update: Try to take LEFT with cost value: {0}".format(traceL))
                    #print("take left agent's sensor state: {0} ".format(agent.state.sensor_states["left_sensor"]))
                    agent.set_state(last_agent_state)
                elif j == 2:
                    obs = sim.step("turn_right")
                    # print("Try to Right")

                    depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

                    # If it is a depth pair, manually normalize into [0, 1]
                    # so that images are always consistent
                    if isdepth:
                        depth_pair = np.clip(depth_pair, 0, 10)
                        # depth_pair /= 10.0

                    isValidR, U_obsR = computeCovariance(agent, depth_pair, targetPositionW)

                    U_postR, traceR = objectiveFun(U_prior, U_obsR)

                    if traceR < traceBest and isValidR:
                        traceBest = traceR
                        bestIndex = j
                        U_post = U_postR
                        #print("Update: Try to take RIGHT with cost value: {0}".format(traceR))

                    #print("take right agent's sensor state: {0} ".format(agent.state.sensor_states["left_sensor"]))
                    agent.set_state(last_agent_state)


            if bestIndex  == -1:
               print("Feature point can not be observed by camera")
               break

            print("In this episode, we finally decide to {0} ".format(dic[bestIndex]))

            obs = sim.step(dic[bestIndex])

            #if camera selce turn left or turn right, set additonal move forward too.
            if bestIndex >= 1:
                obs = sim.step("move_forward")

            #isTrack, obs = rotateCameraToMidpoint(agent, sim, targetPositionW)


            if bestIndex == 0:
                print("In this episode, we finally decide to move forward and cost value is: {0} ".format(traceBest))
            else:
                print("In this episode, we finally decide to take {0} and move forward and cost value is: {1} ".format(
                    dic[bestIndex], traceBest))


            isTrack, _, loc_error = computeTargetPositionOnWorldFrame(agent, targetPositionW)


            input = targetPixelInCurrentCamera(agent, targetPositionW)

            leftImg = obs["left_sensor"]
            if len(leftImg.shape) > 2:
                leftImg = leftImg[..., 0:3][..., ::-1]

            rightImg = obs["right_sensor"]
            if len(rightImg.shape) > 2:
                rightImg = rightImg[..., 0:3][..., ::-1]

            leftIm, rightIm = drawTarget(input, leftImg, rightImg)

            if not isTrack:
                print("Feature point can not be observed by camera")
                # plotResult(y_data, x)
                break
            else:
                y_data = np.append(y_data, loc_error)

            print("Mean Square Error: {0}".format(loc_error))
            #visualize
            stereo_pair = np.concatenate([leftIm, rightIm], axis=1)

            cv2.imshow("stereo_pair", stereo_pair)

            # keystroke = cv2.waitKey(1000)
            # if keystroke == ord("q"):
            #     break

            last_agent_state = agent.state


            ##
            # input = targetPixelInCurrentCamera(agent, targetPositionW)
            # isValid, U_obs = computeCovarianceByPixel(agent, depth_pair, input)
            #
            # R_s_c = np.array([[1, 0, 0], [0, -1, 0],
            #                   [0, 0,
            #                    - 1]])  # c is not left camera here. it is coordinate conresponding to paper but it is same
            # # R_c_s = R_s_c.T
            # R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
            # R_w_c = R_w_s.dot(R_s_c)  # R_w_c
            #
            # delta = sp.Gradient(input, depth, U_obs, R_w_c)
            #
            # # print(delta)
            # UobsNBV, targetPositionCNext = testGradient(agent, targetPositionW, delta)
            # Upost_NBV, traceNBV = objectiveFun(U_prior, UobsNBV)
            # print("traceNBV: {0}".format(traceNBV))
            #
            # nbv_loc = agent.state.sensor_states["left_sensor"].position - getGradientRespectToTranslation(agent,targetPositionW,targetPositionCNext)
            # print("nbv camera poistion in world coordinate: {0}".format(nbv_loc))

            # depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)
            #
            # # If it is a depth pair, manually normalize into [0, 1]
            # # so that images are always consistent
            # if isdepth:
            #     depth_pair = np.clip(depth_pair, 0, 10)
            #
            #
            # R_s_c = np.array([[1, 0, 0], [0, -1, 0],
            #                   [0, 0,
            #                    - 1]])  # c is not left camera here. it is coordinate conresponding to paper but it is same
            # # R_c_s = R_s_c.T
            # R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
            # R_w_c = R_w_s.dot(R_s_c)  # R_w_c
            #
            # input = targetPixelInCurrentCamera(agent, targetPositionW)
            # depth = depth_pair[input[2], input[0]]
            # print("Current Agent's left camera state: {0}".format(agent.state.sensor_states["left_sensor"]))
            # # Gradient of translation of camera
            #
            # delta_camera = sp.GradientofCameraTranslation(input, depth, U_post, R_w_c)
            # # print("input: {0}, depth: {1}, U:{2}, Rwc: {3}".format(input, depth, U_post, R_w_c))
            # # print("delta_camera: {0}".format(delta_camera))
            # # print("delta_camera: {0}".format(delta_camera))
            # NBCameraPosition = nextBestCameraPosition(agent, delta_camera)
            # setAgentPosition(agent, NBCameraPosition)
            # print("Current Agent's left camera state after Gradient Descend: {0}".format(
            #     agent.state.sensor_states["left_sensor"]))
            # # isTrack, _ = rotateCameraToMidpoint(agent, sim, targetPositionW)
            # isTrack, _, loc_error_NB = computeTargetPositionOnWorldFrame(agent, targetPositionW)
            # print("Mean Square Error NB: {0}".format(loc_error_NB))

            # obs = sim.get_sensor_observations()
            # stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
            # if len(stereo_pair.shape) > 2:
            #     stereo_pair = stereo_pair[..., 0:3][..., ::-1]
            # cv2.imshow("stereo_pair", stereo_pair)
            #
            # keystroke = cv2.waitKey(0)
            # if keystroke == ord("q"):
            #     break


            #print("Current Agent's left camera state after Gradient Descend and adjustment: {0}".format(agent.state.sensor_states["left_sensor"]))


            # back to previous agent state
            # setAgentPositionByState(agent, last_agent_state)

            U_prior = U_post
            print("Current Agent's left camera state: {0}".format(agent.state.sensor_states["left_sensor"]))

            i = i+1

        plotResult(y_data, x)

#visualize the target
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

#visualize feature points
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

            if(xL == input[0] and y == input[2]): #target red
                cv2.rectangle(leftIm, (x1, y1), (x2, y2), (0, 0 , 255))
                cv2.circle(leftIm, (xL, y), 2, (0, 0, 255), -1)
            else:
                cv2.rectangle(leftIm, (x1, y1), (x2, y2), (0,255,0))
                cv2.circle(leftIm, (xL, y), 2, (0,255,0), -1)


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
    cur_error = abs(xc_now - cam_baseline/2)
    pre_error = float('inf')
    isLeft = True
    nochange = False
    while pre_error >= cur_error:
        if xc_now > cam_baseline/2:
            sim.step("turn_right")
            isLeft = True
        elif xc_now < cam_baseline/2:
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
        cur_error = abs(xc_rec - cam_baseline/2)

    if isLeft and not nochange:
        print("Rotate camera to right so that z axis of midpoint of baseline go through feature point")
        return True, sim.step("turn_left")
    elif not isLeft and not nochange:
        print("Rotate camera to left so that z axis of midpoint of baseline go through feature point")
        return True, sim.step("turn_right")

def computeTargetPositionOnWorldFrame(agent, targetPositionW_):

    pixel = targetPixelInCurrentCamera(agent, targetPositionW_)
    uL = pixel[0]
    uR = pixel[1]
    v = pixel[2]

    if uL == -1 and uR == -1 and v == -1:
        return False, -1, -1

    cx = width/2
    cy = height/2

    zc = cam_focalLength * cam_baseline / (uL - uR)
    xc = zc * (uL-cx)/cam_focalLength  #xc 3d in cam
    yc = zc * (v - cy)/cam_focalLength #yc

    targetPositionC = np.array([xc, yc, zc])

    # print("targetPosition camera frame: {0}".format(targetPositionC))
    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
    targetPositionS = R_s_c.dot(targetPositionC)

    q = agent.state.sensor_states["left_sensor"].rotation
    R_w_s = quaternion.as_rotation_matrix(q)

    t_w_s = agent.state.sensor_states["left_sensor"].position
    #print("targetPosition sensor frame without translate value: {0}".format(R_w_s.dot(targetPositionS)))
    targetPositionW = R_w_s.dot(targetPositionS) + t_w_s
    # print("targetPosition world value in this episode by taking fixed action: {0}".format(targetPositionW))
    delta_position = targetPositionW - targetPositionW_
    mse_loc = np.linalg.norm(delta_position)
    # print("Mean Square Error: {0}".format(mse_loc))

    return True, targetPositionW, mse_loc

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
    constructPyramid()

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

    agent_state.position = np.array([-1.7926959,  0.11083889, 19.255245])  # [-4.1336217 ,  0.2099186, 12.3464   ] now
    # agent_state.sensor_states["left_sensor"].rotation = np.quaternion(0, 0, 1, 0)
    # agent_state.rotation = np.quaternion(1, 0, 2.45858027483337e-05, 0) 0.0871558338403702, 0, -0.996194660663605, 0

    agent_state.rotation = np.quaternion(1, 0, 0.0, 0)
    agent = sim.initialize_agent(0, agent_state)
    agent.get_state()
    agent.agent_config = agent_config
    return agent.state  # agent.scene_node.transformation_matrix()

def setAgentPosition(agent, NBCameraPosition):
    agent_state = agent.get_state()
    agent_state.position = np.array([NBCameraPosition[0]+cam_baseline/2, NBCameraPosition[1]-1.5, NBCameraPosition[2]])
    agent.set_state(agent_state)
    return agent.state  # agent.scene_node.transformation_matrix()

def setAgentPositionByState(agent, last_agent_state):
    agent.set_state(last_agent_state)
    return agent.state  # agent.scene_node.transformation_matrix()

def targetWordCoordinate(agent, uL, uR, v):
    # targetPosition in left camera coordinate
    cx = width/2
    cy = height/2
    zc = cam_focalLength * cam_baseline / (uL - uR)
    xc = zc * (uL-cx)/cam_focalLength  #xc 3d in cam
    yc = zc * (v - cy)/cam_focalLength #yc

    targetPositionC = np.array([xc, yc, zc])

    print("targetPosition camera frame: {0}".format(targetPositionC))

    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
    targetPositionS = R_s_c.dot(targetPositionC)

    q = agent.state.sensor_states["left_sensor"].rotation
    R_w_s = quaternion.as_rotation_matrix(q)

    t_w_s = agent.state.sensor_states["left_sensor"].position
    #print("targetPosition sensor frame without translate value: {0}".format(R_w_s.dot(targetPositionS)))
    targetPositionW = R_w_s.dot(targetPositionS) + t_w_s
    print("targetPosition world value: {0}".format(targetPositionW))

    return targetPositionW

def targetPixelInCurrentCamera(agent, targetPositionW):

    cx = width / 2
    cy = height / 2

    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
    R_c_s = R_s_c.T

    R_w_s = quaternion.as_rotation_matrix(agent.state.sensor_states["left_sensor"].rotation)

    R_s_w = R_w_s.T
    t_w_s = agent.state.sensor_states["left_sensor"].position
    t_s_w = -1 * R_s_w.dot(t_w_s)

    targetPositionS = R_s_w.dot(targetPositionW) + t_s_w

    targetPositionC = R_c_s.dot(targetPositionS)

    zc = targetPositionC[2]

    xL = (cam_focalLength*targetPositionC[0])/zc + cx
    y = (cam_focalLength*targetPositionC[1])/zc + cy
    xL = round(xL)
    y = round(y)

    xR = xL - ((cam_focalLength * cam_baseline) / zc)
    xR = round(xR)

    if xL < 0 or xL >= width or xR < 0 or xR >=width or y < 0 or y >=height:
        return np.array([-1, -1, -1])
    else:
        pixel = [xL, xR, y]
        return pixel

def project(agent, targetPositionW):
    cx = width/2
    cy = height/2
    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
    R_c_s = R_s_c.T
    R_w_sr = quaternion.as_rotation_matrix(agent.state.sensor_states["right_sensor"].rotation)
    t_w_sr = agent.state.sensor_states["right_sensor"].position
    R_sr_w = R_w_sr.T
    t_sr_w = -1 * R_sr_w.dot(t_w_sr)
    targetPositionS = R_sr_w.dot(targetPositionW) + t_sr_w #sensor
    targetPositionC = R_c_s.dot(targetPositionS)
    xR = (cam_focalLength*targetPositionC[0])/targetPositionC[2] + cx
    y = (cam_focalLength*targetPositionC[1])/targetPositionC[2] + cy
    return np.array([xR, y])


def computeCovariance(agent, depth_pair, targetPositionW):

    pixel = targetPixelInCurrentCamera(agent, targetPositionW)
    # print("target in current frame pixel: {0}". format(pixel))
    cx = width / 2
    cy = height / 2
    xL = pixel[0]
    xR = pixel[1]
    y = pixel[2]

    if xL == -1 and xR == -1 and y ==-1:
        return False, np.identity(3);

    depth = depth_pair[y, xL]
    #depth2 =  cam_baseline*cam_focalLength/(xL-xR)
    level = math.ceil(depth*8/10)
    if level >= 8:
        level = 7

    front_J = cam_baseline / ((xL - xR) ** 2)

    J = np.array([[(cx-xR), (xL-cx), 0],
                  [(cy-y), (y-cy), xL-xR],
                  [-cam_focalLength, cam_focalLength, 0]])

    J = front_J * J

    Q = mvInvLevelSigma2[level] * np.identity(3)
    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)


    R =  R_w_s.dot(R_s_c)

    U_obs = np.linalg.multi_dot([R, J, Q, J.T, R.T])

    return True, U_obs

#init
def computeCovarianceByPixel(agent, depth_pair, input):

    xL = input[0]
    xR = input[1]
    y = input[2]

    cx = width / 2
    cy = height / 2

    depth = depth_pair[y, xL]

    level = int(depth*8/10)

    if level >= 8:
        level = 7

    front_J = cam_baseline / ((xL - xR) ** 2)

    J = np.array([[cx-xR, xL-cx, 0],
                  [cy-y, y-cy, xL - xR],
                  [-cam_focalLength, cam_focalLength, 0]])

    J = front_J * J
    Q = mvInvLevelSigma2[level] * np.identity(3)
    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R = R_w_s.dot(R_s_c)

    U_obs = np.linalg.multi_dot([R, J, Q, J.T, R.T])

    return True, U_obs

#update posterior and computer objective function
def objectiveFun(U_prior, U_obs):

    U_post = (inv(U_prior) + inv(U_obs))
    U_post = inv(U_post)

    return U_post, U_post[0, 0] + U_post[1, 1] + U_post[2, 2]

def objectiveFunInit(U_obs):

    return U_obs[0, 0] + U_obs[1, 1] + U_obs[2, 2]

def testGradient(agent, targetPositionW, delta):
    cx = width / 2
    cy = height / 2

    R_s_c = np.array([[1, 0, 0], [0, -1, 0],
                      [0, 0, - 1]])  # c is not left camera here. it is coordinate conresponding to paper but it is same
    # R_c_s = R_s_c.T
    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R_w_c = R_w_s.dot(R_s_c)  # R_w_c

    R_c_s = R_s_c.T

    R_s_w = R_w_s.T
    t_w_s = agent.state.sensor_states["left_sensor"].position
    t_s_w = -1 * R_s_w.dot(t_w_s)

    targetPositionS = R_s_w.dot(targetPositionW) + t_s_w

    targetPositionC = R_c_s.dot(targetPositionS)

    sum_delta = np.zeros(3)
    #delta =  delta * 0.25/ np.linalg.norm(delta)
    while(np.linalg.norm(sum_delta) < 0.25):
        sum_delta += delta * 0.25
    print("norm sum_delta: {0}".format(np.linalg.norm(sum_delta)))
    print("norm delta: {0}".format(delta))
    print("norm length: {0}".format(np.linalg.norm(t_w_s)))
    targetPositionCNext = targetPositionC - sum_delta

    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
    targetPositionS = R_s_c.dot(targetPositionCNext)

    q = agent.state.sensor_states["left_sensor"].rotation
    R_w_s = quaternion.as_rotation_matrix(q)

    t_w_s = agent.state.sensor_states["left_sensor"].position

    targetPositionW = R_w_s.dot(targetPositionS) + t_w_s
    # print("targetPosition sensor frame value (NBV): {0}".format(R_w_s.dot(targetPositionW)))

    zc = targetPositionCNext[2]

    xL = (cam_focalLength * targetPositionCNext[0]) / zc + cx
    y = (cam_focalLength * targetPositionCNext[1]) / zc + cy
    xL = round(xL)
    y = round(y)

    xR = xL - ((cam_focalLength * cam_baseline) / zc)
    xR = round(xR)

    depth = cam_baseline*cam_focalLength/(xL - xR)

    level = math.ceil(depth * 8 / 10)

    if level >= 8:
        level = 7

    front_J = cam_baseline / ((xL - xR) ** 2)

    J = np.array([[(cx - xR), (xL - cx), 0],
                  [(cy - y), (y - cy), xL - xR],
                  [-cam_focalLength, cam_focalLength, 0]])

    J = front_J * J

    Q = mvInvLevelSigma2[level] * np.identity(3)
    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
    # R_c_s = R_s_c.T
    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R = R_w_s.dot(R_s_c)

    U_obs = np.linalg.multi_dot([R, J, Q, J.T, R.T])

    return  U_obs, targetPositionCNext

def nextBestCameraPosition(agent, delta):

    #need Rwc twc=tws
    R_s_c = np.array([[1, 0, 0], [0, -1, 0],
                       [0, 0, - 1]])  # c is not left camera here. it is coordinate conresponding to paper but it is same

    R_c_s = R_s_c.T

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R_w_c = R_w_s.dot(R_s_c)  # R_w_c
    R_s_w = R_w_s.T
    t_w_s = agent.state.sensor_states["left_sensor"].position

    t_s_w = -R_s_w.dot(t_w_s)

    t_c_w = R_c_s.dot(t_s_w)

    sum_delta = 0.25 * (delta/np.linalg.norm(delta))
    print("sum_delta {0}".format(sum_delta))
    NBCameraPosition = -R_w_c.dot(t_c_w - sum_delta)

    return NBCameraPosition

def getGradientRespectToTranslation(agent, targetPositionW, NBV):
    R_s_c = np.array([[1, 0, 0], [0, -1, 0],
                      [0, 0, - 1]])  # c is not left camera here. it is coordinate conresponding to paper but it is same

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)

    t_w_s = agent.state.sensor_states["left_sensor"].position

    R = R_w_s.dot( R_s_c)

    sum_delta = np.zeros(3)

    delta = 2 * (R.dot(NBV) -targetPositionW + t_w_s)

    while (np.linalg.norm(sum_delta) < 0.25):
        sum_delta += delta

    return sum_delta

#plot the result
def plotResult(y_data, id):

    n = len(y_data)
    x = range(1, n+1)

    fig = plt.figure()
    plt.plot(x, y_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Episode")
    plt.ylabel("Error")
    # plt.yscale('log')
    plt.title("Exp {0}: Position Norm Error".format(id))
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()
    setupAgentwithSensors(display=args.display)
