import numpy as np
import quaternion

import habitat_sim
from numpy.linalg import inv
import math

import AVP_Binding as m

cv2 = None

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
def _render(sim, depth=False):
    agent = sim.get_agent(0)
    print("Target pixel xL: {0}, xR: {1}, y:{2}".format(198, 186, 373))
    targetPositionW = targetWordCoordinate(agent, 198, 186, 373)  #left: 198, 186, 373 right: 884, 871, 311

    print("project on right image pixel: {0}".format(project(agent, targetPositionW)))

    sp = m.SparsePipline("/home/wangweihan/Documents/slam_architecture/ORB_SLAM2/Examples/Stereo/KITTI00-02.yaml")

    print("Current agent's sensor state in {0} Episode: {1} ".format(0, agent.state.sensor_states))
    obs = sim.get_sensor_observations()

    depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

    if depth:
        depth_pair = np.clip(depth_pair, 0, 10)

    input = np.array([198, 186, 373])
    isValid, U_obs = computeCovarianceByPixel(agent, depth_pair, input)
    traceInit = ObjectiveFunInit(U_obs)

    R_s_c = np.array([[1, 0, 0], [0, -1, 0],
                      [0, 0, - 1]])  # c is not left camera here. it is coordinate conresponding to paper but it is same
    # R_c_s = R_s_c.T
    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R_w_c = R_w_s.dot(R_s_c)  # R_w_c

    delta = sp.Gradient(input, depth, U_obs, R_w_c)

    print(delta)
    UobsNBV = TestGradient(agent, targetPositionW, delta)
    Upost_NBV, traceNBV = ObjectiveFun(U_obs, UobsNBV)
    print("traceNBV: {0}".format(traceNBV))
    print("In first episode, the trace of convariance matrix of observation  is: {0} ".format(traceInit))

    last_agent_state = agent.state
    U_prior = U_obs

    # visualize
    # stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
    # if len(stereo_pair.shape) > 2:
    #     stereo_pair = stereo_pair[..., 0:3][..., ::-1]
    #
    # cv2.imshow("stereo_pair", stereo_pair)
    # keystroke = cv2.waitKey(0)


    for i in range(1, 40):
        print("Episode {0}".format(i))

        bestIndex = -1
        U_post = np.identity(3)  # init U_obs
        traceBest = float('inf')
        for j in range(3):
            #print("previous agent's sensor state in {0} Episode: {1} ".format(i, agent.state.sensor_states["left_sensor"]))
            if j == 0:
                obs = sim.step("move_forward")
                # print("Try to move forward")

                depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

                # If it is a depth pair, manually normalize into [0, 1]
                # so that images are always consistent
                if depth:
                    depth_pair = np.clip(depth_pair, 0, 10)
                    # depth_pair /= 10.0


                isValidF, U_obsF = computeCovariance(agent, depth_pair, targetPositionW)

                U_postF, traceF = ObjectiveFun(U_prior, U_obsF)
                # if i == 7:
                #     stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
                #     if len(stereo_pair.shape) > 2:
                #         stereo_pair = stereo_pair[..., 0:3][..., ::-1]
                #     cv2.imshow("stereo_pair", stereo_pair)
                #     keystroke = cv2.waitKey(0)
                #     if keystroke == ord("q"):
                #         break

                if traceF < traceBest and isValidF:
                    traceBest = traceF
                    bestIndex = j
                    U_post = U_postF
                    #print("Update: Try to take FORWARD with cost value: {0}".format(traceF))
                #print("take forward agent's sensor state: {0} ".format(agent.state.sensor_states["left_sensor"]))
                agent.set_state(last_agent_state, False)
            elif j == 1:
                obs = sim.step("turn_left")
                # print("Try to Left")

                depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

                # If it is a depth pair, manually normalize into [0, 1]
                # so that images are always consistent
                if depth:
                    depth_pair = np.clip(depth_pair, 0, 10)
                    # depth_pair /= 10.0

                isValidL, U_obsL = computeCovariance(agent, depth_pair, targetPositionW)

                U_postL, traceL = ObjectiveFun(U_prior, U_obsL)
                # stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
                # if len(stereo_pair.shape) > 2:
                #     stereo_pair = stereo_pair[..., 0:3][..., ::-1]
                # if i == 7:
                #     cv2.imshow("stereo_pair", stereo_pair)
                #     keystroke = cv2.waitKey(0)
                #     if keystroke == ord("q"):
                #         break
                if traceL < traceBest and isValidL:
                    traceBest = traceL
                    bestIndex = j
                    U_post = U_postL
                    #print("Update: Try to take LEFT with cost value: {0}".format(traceL))
                #print("take left agent's sensor state: {0} ".format(agent.state.sensor_states["left_sensor"]))
                agent.set_state(last_agent_state, False)
            elif j == 2:
                obs = sim.step("turn_right")
                # print("Try to Right")

                depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

                # If it is a depth pair, manually normalize into [0, 1]
                # so that images are always consistent
                if depth:
                    depth_pair = np.clip(depth_pair, 0, 10)
                    # depth_pair /= 10.0

                isValidR, U_obsR = computeCovariance(agent, depth_pair, targetPositionW)

                U_postR, traceR = ObjectiveFun(U_prior, U_obsR)

                if traceR < traceBest and isValidR:
                    traceBest = traceR
                    bestIndex = j
                    U_post = U_postR
                    #print("Update: Try to take RIGHT with cost value: {0}".format(traceR))

                #print("take right agent's sensor state: {0} ".format(agent.state.sensor_states["left_sensor"]))
                agent.set_state(last_agent_state, False)

        obs = sim.step(dic[bestIndex])
        if bestIndex >= 1:
            sim.step("move_forward")

        if bestIndex == 0:
            print("In this episode, we finally decide to move forward and cost value is: {0} ".format(traceBest))
        else:
            print("In this episode, we finally decide to take {0} and move forward and cost value is: {1} ".format(
                dic[bestIndex], traceBest))

        #visualize
        stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
        if len(stereo_pair.shape) > 2:
            stereo_pair = stereo_pair[..., 0:3][..., ::-1]
        cv2.imshow("stereo_pair", stereo_pair)
        keystroke = cv2.waitKey(1000)
        if keystroke == ord("q"):
            break
        last_agent_state = agent.state
        print("Current agent's sensor state in {0} Episode: {1} ".format(i, agent.state.sensor_states["left_sensor"]))
        U_prior = U_post


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
    _render(sim, depth=True)

def place_agent(sim, agent_config):
    # place our agent in the scene
    agent_state = habitat_sim.agent.AgentState()

    # agent_state.position = [-1.7926959 ,  0.11083889, 19.255245 ] inital

    agent_state.position = [-1.7926959 ,  0.11083889, 19.255245 ]  # [-4.1336217 ,  0.2099186, 12.3464   ] now
    # agent_state.sensor_states["left_sensor"].rotation = np.quaternion(0, 0, 1, 0)
    # agent_state.rotation = np.quaternion(1, 0, 2.45858027483337e-05, 0) 0.0871558338403702, 0, -0.996194660663605, 0

    agent_state.rotation = np.quaternion(1, 0, 0.0, 0)
    agent = sim.initialize_agent(0, agent_state)
    agent.get_state()
    agent.agent_config = agent_config
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
    xL = int(xL)
    y = int(y)

    xR = xL - ((cam_focalLength * cam_baseline) / zc)
    xR = int(xR)

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
    # R_c_s = R_s_c.T
    # rotation = np.quaternion(1, 0, 0, 0)
    # R_w_s = quaternion.as_rotation_matrix(rotation)
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
    front_J = cam_baseline / ((xL - xR) ** 2)

    J = np.array([[cx-xR, xL-cx, 0],
                  [cy-y, y-cy, xL - xR],
                  [-cam_focalLength, cam_focalLength, 0]])

    J = front_J * J
    Q = mvInvLevelSigma2[level] * np.identity(3)
    R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
    # R_c_s = R_s_c.T
    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R = R_w_s.dot(R_s_c)

    U_obs = np.linalg.multi_dot([R, J, Q, J.T, R.T])

    return True, U_obs

#update posterior and computer objective function
def ObjectiveFun(U_prior, U_obs):

    U_post = (inv(U_prior) + inv(U_obs))
    U_post = inv(U_post)

    return U_post, U_post[0, 0] + U_post[1, 1] + U_post[2, 2]

def ObjectiveFunInit(U_obs):

    return U_obs[0, 0] + U_obs[1, 1] + U_obs[2, 2]

def TestGradient(agent, targetPositionW, delta):
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

    while(np.linalg.norm(sum_delta) < 0.25):
        sum_delta += delta

    targetPositionCNext = targetPositionC - sum_delta

    zc = targetPositionCNext[2]

    xL = (cam_focalLength * targetPositionCNext[0]) / zc + cx
    y = (cam_focalLength * targetPositionCNext[1]) / zc + cy
    xL = int(xL)
    y = int(y)

    xR = xL - ((cam_focalLength * cam_baseline) / zc)
    xR = int(xR)

    depth = cam_baseline*cam_focalLength/(xL - xR)

    level = math.ceil(depth * 8 / 10)
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

    return  U_obs
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()
    setupAgentwithSensors(display=args.display)
