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

def computeCameraTranslation(agent,targetCurrentPosition, targetNextPosition):

    # R_w_s = quaternion.as_rotation_matrix(agent.state.sensor_states["left_sensor"].rotation)
    # targetC = targetNextPosition + tleftcamera_cyclopean
    # Rwc = R_w_s.dot(R_s_c)
    # tws = agent.state.sensor_states["left_sensor"].position
    #
    # targetW = Rwc.dot(targetC) + tws
    #
    # twcy = tws - Rwc.dot(tcyclopean_leftcamera)
    #
    # delta_twcy = 2 * (Rwc.dot(targetNextPosition) + twcy - targetW_)
    #
    # delta_twcy = delta_twcy / np.linalg.norm(delta_twcy)
    #
    # nextBestCameraPositionCy = twcy - 0.25 * delta_twcy
    #
    # print("delta_tws: {0}, norm of delta_tws: {1}".format(twcy, np.linalg.norm(twcy)))
    # print("nextBestCameraPostion: {0}".format(nextBestCameraPositionCy))
    # return nextBestCameraPositionCy

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


def computeCameraRotation(agent, targetCurrentPosition, targetNextPosition, R):
    twc1 = agent.state.sensor_states["left_sensor"].position  # twc at tk-1
    Rws1 = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    Rwc1 = Rws1.dot(R_s_c)
    Rwcy1 = Rwc1  # R(tk-1)
    twcy1 = twc1 - Rwcy1.dot(tcyclopean_leftcamera)  #twcy at tk-1

    r_star = twcy1 + Rwcy1.dot(targetNextPosition - targetCurrentPosition)
    r_star = r_star[np.newaxis].T

    targetW = Rwcy1.dot(targetCurrentPosition) + twcy1
    targetW = targetW[np.newaxis].T

    z_hat = (targetW - r_star)/np.linalg.norm(targetW - r_star)

    e3 = np.array([0, 0, 1])[np.newaxis].T

    R1 = np.linalg.multi_dot([R.T, z_hat, (R.T.dot(z_hat)-e3).T])
    R2 = np.linalg.multi_dot([R.T.dot(z_hat) - e3, z_hat.T, R])
    delta_R = R1 - R2

    R = R.dot(delta_R)

    R = R + delta_R

    return R

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
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    # fig.savefig('Experiment_{0}_traj.png'.format(exId+1), dpi=fig.dpi)
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
        # stereo_pair = np.concatenate([leftImg, rightImg], axis=1)
        #
        # cv2.imshow("stereo_pair", stereo_pair)
        # cv2.waitKey(1000)

        # time k = 0
        x_nbv = []
        z_nbv = []
        agent.set_state(init_state)

        # new feature point in pixel coordinate
        input = np.array([inputs[exId][0], inputs[exId][1], inputs[exId][2]])
        # test
        input = np.array([198, 186, 373])
        obs = sim.get_sensor_observations()
        # stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
        # if len(stereo_pair.shape) > 2:
        #     stereo_pair = stereo_pair[..., 0:3][..., ::-1]
        # cv2.imshow("stereo_pair", stereo_pair)
        #
        # keystroke = cv2.waitKey(1000)
        # if keystroke == ord("q"):
        #     break
        targetPositionW = targetWordCoordinate(agent, 198, 186, 373) #917, 905, 322   198, 186, 373
        # targetPositionW = targetWordCoordinate(agent, input[0], input[1], input[2])  #left: 198, 186, 373 right: 917,905,322 437, 431, 293
        # input_withoutOffset = np.array([input[0]-cx, input[1]-cx, input[2]-cy])
        # targetPositionCy = targetPositionIncycolpean(input_withoutOffset)
        # Rws = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
        # Rwc = Rws.dot(R_s_c)
        # Rwcy = Rwc  # R(tk-1)
        #
        # R = rotateCameraToMidpoint(targetPositionCy)
        # t = R.dot(targetPositionCy)
        # print(t)
        # Rwcy_next = Rwcy.dot(R.T)
        # setAgentRotation(agent, Rwcy_next)

        print("project on right image pixel: {0}".format(project(agent, targetPositionW)))

        depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

        depth_pair = np.clip(depth_pair, 0, 10)



        # depth of target
        depth = depth_pair[int(input[2]), int(input[0])]

        input_withoutOffset = np.array([input[0]-cx, input[1]-cx, input[2]-cy])
        J = makeJacobian(input_withoutOffset)

        Q = makeQ(depth)

        U_obs = computeCovarianceByPixel(agent, J, Q)

        # x_nbv.append(agent.state.position[0])
        # z_nbv.append(agent.state.position[2])


        # Next target and camera position, timestep k = 1
        U_prior = U_obs

        R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
        R_w_c = R_w_s.dot(R_s_c)  # R_w_c

        # Uk+1|k, Uobs
        delta = sp.GradientForTarget(input_withoutOffset, depth, U_prior, R_w_c)

        print("delta 0: {0}".format(delta/np.linalg.norm(delta)))
        #predict next best target position and camera position
        targetPositionCy, targetPositionCyNext = getNextBestPositionofFeature(input_withoutOffset, delta)
        print("targetPositionCNext in cyclopean coordinate: {0}".format(targetPositionCyNext))
        twc = agent.state.sensor_states["left_sensor"].position  # twc at tk-1
        Rws = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
        Rwc = Rws.dot(R_s_c)
        Rwcy = Rwc  # R(tk-1)

        nextBestCameraPostionIncy = computeCameraTranslation(agent, targetPositionCy, targetPositionCyNext)
        R = rotateCameraToMidpoint(targetPositionCyNext)
        t = R.dot(targetPositionCyNext)
        print(t)
        Rwcy_rectify = Rwcy.dot(R.T)
        # print("Rwcy_next : {0}".format(Rwcy_rectify))


        x_nbv.append(nextBestCameraPostionIncy[0])
        z_nbv.append(nextBestCameraPostionIncy[2])

        # update camera position

        # nextBestCameraPostionIncy = np.array([-1.7926959, 1.6196585, 19.005245])
        setAgentPosition(agent, nextBestCameraPostionIncy)
        setAgentRotation(agent, Rwcy_rectify)



        #viz
        # obs = sim.get_sensor_observations()
        # stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
        # if len(stereo_pair.shape) > 2:
        #     stereo_pair = stereo_pair[..., 0:3][..., ::-1]
        # cv2.imshow("stereo_pair", stereo_pair)
        #
        # keystroke = cv2.waitKey(1000)
        # if keystroke == ord("q"):
        #     break

        # obs = sim.step("turn_right")
        # obs = sim.step("turn_right")
        # obs = sim.step("turn_right")



        # chose fix move and do gradient descent
        for i in range(1, 200):  # np.sqrt(loc_error) > 1e-3:
            print("+++++++++++++Step {0}++++++++".format(i))
            print("Current Agent's state: {0}".format(agent.state))
            input = targetPixelInCurrentCamera(agent, targetPositionW)
            # print(input)

            if not isObserved(input):
                print("Feature point can not be observed by camera")
                break
            else:
                R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
                R_w_c = R_w_s.dot(R_s_c)

                err = computeLocError(agent, input, targetPositionW, R_w_c)
                error_array.append(err)

                dis = computeDistanceFromTarget(agent, targetPositionW)
                dis_array.append(dis)
                # print("loc error: {0}".format(err))

                depth_pair = np.concatenate([obs["left_sensor_depth"], obs["right_sensor_depth"]], axis=1)

                depth_pair = np.clip(depth_pair, 0, 10)

                U_obs = computeObsCovariance(agent, depth_pair, targetPositionW)

                U_post, trace = objectiveFun(U_prior, U_obs)
                trace_array.append(trace)

                print("Current Agent's trace: {0}".format(trace))

                #visualize
                obs = sim.get_sensor_observations()
                stereo_pair = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
                if len(stereo_pair.shape) > 2:
                    stereo_pair = stereo_pair[..., 0:3][..., ::-1]
                cv2.imshow("stereo_pair", stereo_pair)

                keystroke = cv2.waitKey(0)
                if keystroke == ord("q"):
                    break

                # Predict next best position of target and camera at time k+1 base on current observation at time k
                depth = depth_pair[int(input[2]), int(input[0])]
                input_withoutOffset = np.array([input[0] - cx, input[1] - cx, input[2] - cy])
                delta = sp.GradientForTarget(input_withoutOffset, depth, U_prior, R_w_c)

                print("delta 0: {0}".format(delta/np.linalg.norm(delta)))
                targetPositionCy, targetPositionCyNext = getNextBestPositionofFeature(input_withoutOffset, delta)
                print("targetPositionCNext in cyclopean coordinate: {0}".format(targetPositionCyNext))

                # twc = agent.state.sensor_states["left_sensor"].position  # twc at tk-1
                Rws = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
                Rwc = Rws.dot(R_s_c)
                Rwcy = Rwc  # R(tk-1)

                nextBestCameraPostionIncy = computeCameraTranslation(agent, targetPositionCy, targetPositionCyNext)
                R = rotateCameraToMidpoint(targetPositionCyNext) #Rck-1ck
                Rwcy_rectify = Rwcy.dot(R.T)
                # # nextBestCameraRotation = computeCameraRotation(agent, targetPositionCy, targetPositionCyNext, Rwcy)
                x_nbv.append(nextBestCameraPostionIncy[0])
                z_nbv.append(nextBestCameraPostionIncy[2])

                # update camera position
                setAgentPosition(agent, nextBestCameraPostionIncy)

                # setAgentRotation(agent, Rwcy_rectify)
                # print("Current Agent's state after Gradient Descend with rotation: {0}".format(agent.state))

                U_prior = U_post


        # exId = 2
        # drawTrajectory(x_nbv, z_nbv, targetPositionW, exId)
        # plotLocError(error_array, exId)
        plotTrace(trace_array, exId)
        # if exId == 0:
        # # y2 = [2.305539769016563, 0.8263557457755599, 0.3540837839763306, 0.1720042645012877, 0.0916494988566281, 0.04586429366700652, 0.02686729433471213, 0.016794677570824344, 0.0108787178831339, 0.007213834661898134, 0.004384054062706234, 0.003957583555667254, 0.0030890447312009483, 0.0021548619776846, 0.001545658994013445, 0.0011246652873587344, 0.0008253481522115234, 0.0006085378702942314, 0.0005463067721779868, 0.0004394990016034706, 0.0003210266687729331, 0.00023892905408671853, 0.0001793587936998455, 0.00016173476842999915, 0.00013116489759876609, 0.00010406669250640404, 8.08817227216705e-05, 7.331521734687509e-05, 5.952796051729625e-05, 4.3472233225090505e-05, 3.992862310285069e-05, 3.0788271198775396e-05, 2.3263077194906992e-05, 2.0835969283592946e-05, 1.5944681074746935e-05, 1.1667698267310537e-05, 1.0428658768914531e-05, 9.219340841509037e-06]
        #     y2 = [0.4132777650437498, 0.10900206579852059, 0.04304723661158948, 0.019790866004557914, 0.0173899784618795, 0.00919454209881029, 0.00533416680802362, 0.0032815320614890705, 0.0021013983841605376, 0.0012383706309839274, 0.001108526709186729, 0.0008554347544238323, 0.0005879980360637433, 0.00033688467251972125, 0.0002289166383253578, 0.00016953465650786006, 0.0001471871476948809, 0.0001359497401639505, 0.00013130520676955888, 0.00012874834276644932, 0.00012702559341848705, 0.0001243358277666199, 0.00012220607912159703, 0.0001208595973945776, 0.0001196949488753961, 0.0001179960584451521, 0.00011556615584759931, 0.00011156544177030193]
        #     dis2 = [1.3732700395566712e-14, 5.197930934883577e-15, 2.1354411886641158e-14, 1.1682163181672949e-14, 2.3228879851839006e-14, 1.6213818077951355e-14, 2.5510982866352577e-15, 1.4073144318774333e-14, 6.936895214610187e-15, 1.3617327804437774e-14, 6.1894311297400655e-15, 9.272854936269104e-15, 1.0168091383860044e-14, 4.440892098500626e-16, 9.155133597044475e-16, 6.7678088263037544e-15, 7.49708873476879e-15, 3.4684476073050936e-15, 1.0583867371683362e-14, 5.895691955682626e-15, 5.5688503603976495e-15, 1.4895204919483639e-15, 2.0350724194510405e-15, 8.005932084973442e-16, 5.407139474782739e-15, 8.251579053847122e-15, 6.481268641478987e-15]
        # elif exId == 1:
        #     y2 = [4.583942360418052, 2.81094956129878, 1.9234202203092372, 1.3916583255859225, 1.0391746081045068, 0.7903424350935518, 0.6073073415725962, 0.4689573180627072, 0.36251548033588216, 0.27972601649963214, 0.21495610641583365, 0.16418292409612206, 0.12442280046862852, 0.0933949808936162, 0.07069924902299905, 0.05271217221164481, 0.04160370287727804, 0.03443809801046807, 0.028259962818351494, 0.013615429442757454, 0.0020869354161929705, 0.0015551604505555235]
        #     # dis2 = [2.582050345126043e-14, 1.7373837509653234e-14, 1.2932072993408848e-14, 2.182033134733423e-14,
        #     #         1.3455701531316242e-14, 8.95090418262362e-16, 8.926082647349967e-15, 4.010656666373001e-15,
        #     #         9.551144859011592e-15, 1.1102230246251565e-16, 9.483150488903745e-15, 8.074153716986702e-15, 0.0,
        #     #         3.233018248352212e-15, 3.978256139440565e-15, 8.95090418262362e-16, 3.2953244754398602e-15,
        #     #         4.453364592678439e-15, 4.463041323674983e-15, 2.673771110915334e-15, 4.551914400963142e-15,
        #     #         3.2709212810858e-15, 2.175583928816829e-15, 9.155133597044475e-16, 8.95090418262362e-16,
        #     #         3.2709212810858e-15, 3.3232593448441795e-15, 9.155133597044475e-16, 2.7217452008024307e-15,
        #     #         2.7012892057857038e-15, 9.485749680535094e-16, 9.930136612989092e-16, 2.0137621733714643e-15,
        #     #         9.485749680535094e-16, 1.9891280697202825e-15, 1.1102230246251565e-16, 8.881784197001252e-16]
        #     dis2 = [1.2468155522574926e-14, 8.992121150493565e-15, 1.8444410139024814e-15, 8.895651159694986e-15, 5.407139474782739e-15, 1.8444410139024814e-15, 1.9100999153570945e-15, 1.9860273225978185e-15, 1.831026719408895e-15, 5.352150186685534e-15, 5.338314359524835e-15, 1.790180836524724e-15, 1.7763568394002505e-15, 1.831026719408895e-15, 7.242875823676482e-15, 3.580361673049448e-15, 5.407139474782739e-15, 3.66205343881779e-15, 5.497566137053743e-15, 1.790180836524724e-15, 4.440892098500626e-16]
        #
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

def setAgentPosition(agent, nextBestCameraPostionIncy):

    R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
    R_w_c = R_w_s.dot(R_s_c)  # R_w_c

    nextBestCameraPosition = nextBestCameraPostionIncy + R_w_c.dot(tcyclopean_leftcamera)

    agent_state = agent.get_state()
    # y = agent_state.position[1]c
    y = nextBestCameraPostionIncy[1]
    agent_state.position = np.array([nextBestCameraPosition[0]+cam_baseline/2, y-1.5, nextBestCameraPosition[2]])
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


def computeTargetPositionOnWorldFrame(agent, targetPositionW_):

    pixels = targetPixelInCurrentCamera(agent, targetPositionW_)

    if isObserved(pixels):
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


def project(agent, targetPositionW):
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

    pixels_withoutOffset = np.array([pixels[0] - cx, pixels[1] - cx, pixels[2] - cy])

    J = makeJacobian(pixels_withoutOffset)
    Q = makeQ(depth_pair[int(pixels[2]), int(pixels[0])])

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




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()
    setupAgentwithSensors(display=args.display)
