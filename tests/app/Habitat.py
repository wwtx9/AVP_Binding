import AVP_Binding
import numpy as np
import quaternion
import habitat_sim
import copy
from numpy.linalg import inv
import math
import FeatureTracker
import Frame
import MapPoint
import SparsePipeline
import Plot
import TestFun

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"

cam_baseline = 0.2
cam_focalLength = 450

cx = 480
cy = 320
height = 640
width = 960

dic = {0: "move_forward", 1: "turn_left", 2: "turn_right"}

R_s_c = np.array([[1, 0, 0], [0, -1, 0], [0, 0, - 1]])
tcyclopean_leftcamera = np.array([-cam_baseline/2, 0, 0])
tleftcamera_cyclopean = np.array([cam_baseline/2, 0, 0])
class KeyPoint:
    def __init__(self, x, y, size, angle, response, octave, class_id):
        self.x = x
        self.y = y
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id

def drawFeaturePoints(leftOld, leftIm, LastFrame, CurrentFrame):
    # trackedKpIndexs = CurrentFrame.originTrackedKpIndexs
    trackedKpIndexs = CurrentFrame.preTrackedKpIndexs
    color = np.random.randint(0, 255, (len(trackedKpIndexs), 3))
    for i in range(len(trackedKpIndexs)):
        c, d = LastFrame.Kps[trackedKpIndexs[i]]
        a, b = CurrentFrame.Kps[i]

        # if trackedKpIndexs[i] == 965:
        #      cv2.circle(leftIm, (a, b), 5, (0, 0, 255), -1)
        #      cv2.circle(leftOld, (c, d), 5, (0, 0, 255), -1)
        # else:
        #      cv2.circle(leftIm, (a, b), 5, color[i].tolist(), -1)
        #      cv2.circle(leftOld, (c, d), 5, color[i].tolist(), -1)
        cv2.circle(leftIm, (a, b), 5, color[i].tolist(), -1)
        cv2.circle(leftOld, (c, d), 5, color[i].tolist(), -1)


    # return frameOld, frame
def drawFeaurePointsWithActivePoint(leftOld, leftIm, LastFrame, CurrentFrame, activeIndex):
    trackedKpIndexs = CurrentFrame.preTrackedKpIndexs
    color = np.random.randint(0, 255, (len(trackedKpIndexs), 3))
    for i in range(len(trackedKpIndexs)):
        c, d = LastFrame.Kps[trackedKpIndexs[i]]
        a, b = CurrentFrame.Kps[i]
        if i == activeIndex:
            cv2.circle(leftOld, (c, d), 7, (0, 0, 255), -1)
            cv2.circle(leftIm, (a, b), 7, (0, 0, 255), -1)
        # else:
        #     cv2.circle(leftOld, (c, d), 5, color[i].tolist(), -1)
        #     cv2.circle(leftIm, (a, b), 5, color[i].tolist(), -1)

def drawTrackedKeyPoint(leftIm, rightIm, Kps, KpRights, curTrackedKpIndex):
    uL = Kps[curTrackedKpIndex, 0]
    v = Kps[curTrackedKpIndex, 1]
    uR = KpRights[curTrackedKpIndex, 0]

    cv2.circle(leftIm, (uL, v), 3, (0, 0, 255), -1)
    cv2.circle(rightIm, (uR, v), 3, (0, 0, 255), -1)

def drawLeftRightFeaturePointsWithActivePoint(leftIm, rightIm, Kps, KpRights, activeIndex):
    color = np.random.randint(0, 255, (len(Kps), 3))
    for i in range(len(Kps)):
        uL = Kps[i, 0]
        v = Kps[i, 1]
        uR = KpRights[i, 0]

        if i == activeIndex:
            r = 5
            x1 = round(uL - r)
            y1 = round(v - r)

            x2 = round(uL + r)
            y2 = round(v + r)
            cv2.rectangle(leftIm, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.circle(leftIm, (uL, v), 3, (0, 0, 255), -1)

            xR1 = round(uR - r)
            yR1 = round(v - r)

            xR2 = round(uR + r)
            yR2 = round(v + r)

            cv2.rectangle(rightIm, (xR1, yR1), (xR2, yR2), (0, 0, 255))
            cv2.circle(rightIm, (uR, v), 3, (0, 0, 255), -1)
        else:
            cv2.circle(leftIm, (uL, v), 4, color[i].tolist(), -1)
            cv2.circle(rightIm, (uR, v), 4, color[i].tolist(), -1)


def drawLeftRightFeaturePoints(leftIm, rightIm, cur_pt, cur_pt_R):
    good_L = cur_pt.reshape(-1, 2)
    good_R = cur_pt_R.reshape(-1, 2)
    assert len(cur_pt) == len(cur_pt_R)
    color = np.random.randint(0, 255, (len(good_L), 3))
    for i, (L, R) in enumerate(zip(good_L, good_R)):
        a, b = L.ravel()
        c, d = R.ravel()

        # if 680 >= a >= 670 and 490 >= b >= 480:
        #     print(i)
        #     cv2.circle(leftIm, (a, b), 7, color[i].tolist(), -1)
        #     cv2.circle(rightIm, (c, d), 7, color[i].tolist(), -1)
        # else:
        cv2.circle(leftIm, (a, b), 5, color[i].tolist(), -1)
        cv2.circle(rightIm, (c, d), 5, color[i].tolist(), -1)


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
        "/home/wangweihan/PycharmProjects/AVP_Python/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb")  # data/scene_datasets/habitat-test-scenes/skokloster-castle.glb

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
    _render(sim)

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

def _render(sim):
    # init and extract orb feature at beginning image
    agent = sim.get_agent(0)
    # Start our system
    sy = AVP_Binding.System(
        "/home/wangweihan/PycharmProjects/AVP_Python/habitat.yaml")
    init_state = agent.state
    print("init agent position: {0}".format(init_state.position))

    for exId in range(1, 2):
        #print("++++++++Experiment {0}+++++++".format(exId))

        #test and plot
        avg_loc_error_array = []
        trace_array = []



        #result
        # loc_error_array = []
        mapPoint_array = []

        for step in range(0, 25):
            print("++++++++Current step: {0}++++++".format(step))
            print(agent.state)
            obs = sim.get_sensor_observations()
            leftImgOrg = obs["left_sensor"]
            if len(leftImgOrg.shape) > 2:
                # change image from rgb to bgr
                leftImgOrg = leftImgOrg[..., 0:3][..., ::-1]
            leftImgOrg = np.array(leftImgOrg)
            rightImgOrg = obs["right_sensor"]
            if len(rightImgOrg.shape) > 2:
                # change image from rgb to bgr
                rightImgOrg = rightImgOrg[..., 0:3][..., ::-1]
            rightImgOrg = np.array(rightImgOrg)

            #Convert to gray image
            frame_gray = cv2.cvtColor(leftImgOrg, cv2.COLOR_BGR2GRAY)
            frame_R_gray = cv2.cvtColor(rightImgOrg, cv2.COLOR_BGR2GRAY)

            # Extract feature point
            if step == 0:
                preTrackedKpIndexs = []
                sy.ProcessingStereo(leftImgOrg, rightImgOrg, step)
                sp = sy.mpSparsePipline
                CurrentFrame_cpp = sp.mCurrentFrame
                npuRights = np.array(CurrentFrame_cpp.mvuRight)
                numKps = len(npuRights)
                npKps_array = np.array([KeyPoint(-1, -1, -1, -1, -1, -1, -1) for _ in range(numKps)])
                for kpId in range(numKps):
                    kp_cpp = CurrentFrame_cpp.mvKPs[kpId]
                    npKps_array[kpId] = KeyPoint(np.float32(kp_cpp.x), np.float32(kp_cpp.y), kp_cpp.size, kp_cpp.angle,
                                                 kp_cpp.response, kp_cpp.octave, kp_cpp.class_id)

            if step == 0:
                cur_pt = []
                for kpId in range(numKps):
                    cur_pt.append([npKps_array[kpId].x, npKps_array[kpId].y])
                #First left image do not KLT tracking
                cur_pt = np.array(cur_pt)
                cur_pt = cur_pt.reshape(-1, 1, 2)
                st = np.ones(numKps)
                st = st.astype(np.uint8)
                st = st.reshape(numKps, 1)
            else:
                #KLT Tracking
                #Track keypoint from previous left image
                preTrackedKpIndexs = []
                if len(pre_pt) >= 1:
                    old_gray = cv2.cvtColor(leftOld, cv2.COLOR_BGR2GRAY)
                    cur_pt, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, pre_pt, None, **FeatureTracker.lk_params)
                    # print("pre_pt id: {0}".format(id(pre_pt)))
                    # print("cur_pt id: {0}".format(id(cur_pt)))
                    if len(cur_pt) >= 1:
                        #back KTL to check
                        reverse_pt, reverse_st, reverse_err = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, cur_pt, None, **FeatureTracker.lk_params)
                        # print("reverse_pt id: {0}".format(id(reverse_pt)))
                        #Select good points
                        if st is not None:
                            for st_index in range(len(st)):

                                if st[st_index] == 1 and reverse_st[st_index] == 1 and FeatureTracker.distance(pre_pt[st_index], reverse_pt[st_index]) <= 3:
                                    st[st_index] = 1
                                else:
                                    st[st_index] = 0

                        #check if kp is out of range or not
                        for pt_index in range(len(cur_pt)):
                            if st[pt_index] == 1 and FeatureTracker.inBorder(cur_pt[pt_index]) is False:
                                st[pt_index] = 0


                        #store previous tracked keypoint index
                        for st_index in range(len(st)):
                            if st[st_index] == 1:
                                preTrackedKpIndexs.append(st_index)

                        good_new = cur_pt[st == 1]
                        st = st[st == 1]
                        st = st.reshape(-1, 1)
                        cur_pt = good_new.reshape(-1, 1, 2)
                        # print("good_new id: {0}".format(id(good_new)))
                        print("current track keypoint number: {0}".format(len(good_new)))
                        if len(good_new) == 0:
                            print("Can not track any points")
                            break
                        # #Test on visulaization
                        # #good_old_v = pre_pt[st == 1]
                        # #pre_pt_v = good_old_v.reshape(-1, 1, 2)
                        #
                        # # if len(cur_pt) >= 8:
                        # #     #Add Fundamental constraint to remove outlier
                        # #     F, mask = cv2.findFundamentalMat(pre_pt, cur_pt, cv2.FM_RANSAC, 1.0, 0.99, 200)
                        # #     if mask is not None:
                        # #         good_newWithF = cur_pt[mask.ravel() == 1]
                        # #         good_oldWithF = pre_pt[mask.ravel() == 1]
                        # #         pre_pt = good_oldWithF.reshape(-1, 1, 2)
                        # #         cur_pt = good_newWithF.reshape(-1, 1, 2)
                        # #     else:
                        # #         print("track unsuccessfully add more keypoints")
                        #
                        # print("current track keypoint number after Fundamental constraint: {0}".format(len(cur_pt)))
                        # #visulize
                        # if len(cur_pt) > 0:
                        #     leftOldDraw = leftOld.copy()
                        #     leftImgDraw = leftImgOrg.copy()
                        #     drawFeaturePoints(leftOldDraw, leftImgDraw, pre_pt, cur_pt)
                        #     print("Enter any key")
                        #     stereo_pair = np.concatenate([leftOldDraw, leftImgDraw], axis=1)
                        #     #tereo_pair = np.concatenate([leftOld, leftImgOrg], axis=1)
                        #     cv2.imshow("stereo_pair", stereo_pair)
                        #     keystroke = cv2.waitKey(0)
                        #     if keystroke == ord("q"):
                        #         break
                        # #end test

            #Track keypoints in the right image
            cur_pt_R, st_R, err_R = cv2.calcOpticalFlowPyrLK(frame_gray, frame_R_gray, cur_pt, None, **FeatureTracker.lk_params)
            #print("cur_pt_R id: {0}".format(id(cur_pt_R)))

            #Reversed Check
            if len(cur_pt_R) >= 1:
                # back KTL to check left image
                reverse_pt_L, reverse_st_L, reverse_err_L = cv2.calcOpticalFlowPyrLK(frame_R_gray, frame_gray, cur_pt_R, None, **FeatureTracker.lk_params)
                #print("reverse_pt_L id: {0}".format(id(reverse_pt_L)))
                # Select good points
                if st_R is not None:
                    for st_index in range(len(st_R)):
                        if st_R[st_index] == 1 and reverse_st_L[st_index] == 1 and FeatureTracker.checkDisparity(cur_pt[st_index], reverse_pt_L[st_index]) <= 3:
                            st_R[st_index] = 1
                        else:
                            # did not find right point
                            st_R[st_index] = 0


            #Unify st and st_R
            assert len(st) == len(st_R)
            for st_index in range(len(st_R)):
                if st_R[st_index] != 1:
                    st_R[st_index] = 0
                    st[st_index] = 0
                else:
                    st[st_index] = 1
                    st_R[st_index] = 1

            #remove occluded points
            if step > 0:
                deleted_indexs = []
                for st_index in range(len(st)):
                    if st[st_index] != 1:
                        deleted_indexs.append(st_index)
                preTrackedKpIndexs = np.delete(preTrackedKpIndexs, deleted_indexs)


            good_new = cur_pt[st == 1]
            cur_pt = good_new.reshape(-1, 1, 2)
            good_new_R = cur_pt_R[st_R == 1]
            cur_pt_R = good_new_R.reshape(-1, 1, 2)

            if step == 0:
                preTrackedKpIndexs = np.array(range(0, len(cur_pt)))
                originTrackedKpIndexs = np.array(range(0, len(cur_pt)))


            #VIS show: left-right
            # leftDraw = leftImgOrg.copy()
            # rightDraw = rightImgOrg.copy()
            # drawLeftRightFeaturePoints(leftDraw, rightDraw, cur_pt, cur_pt_R)
            # stereo_pair = np.concatenate([leftDraw, rightDraw], axis=1)
            #
            # cv2.imshow("stereo_pair", stereo_pair)
            # keystroke = cv2.waitKey(0)
            # if keystroke == ord("q"):
            #     break

            #Construct CurrentFrame
            current_trackedKp_N = len(cur_pt)
            npKeyPointsObsCovariance_array = np.zeros((current_trackedKp_N, 3, 3), dtype="float32")
            npKeyPointsPostCovariance_array = np.zeros((current_trackedKp_N, 3, 3), dtype="float32")
            Kps = cur_pt.reshape(-1, 2)
            KpRights = cur_pt_R.reshape(-1, 2)

            #add father union find
            if step > 0:
                originTrackedKpIndexs = np.array(range(0, len(preTrackedKpIndexs)))
                for cur_Kp_index in range(len(preTrackedKpIndexs)):
                    trackedIndex = preTrackedKpIndexs[cur_Kp_index]
                    originTrackedKpIndexs[cur_Kp_index] = LastFrame.originTrackedKpIndexs[trackedIndex]

            CurrentFrame = Frame.Frame(step, current_trackedKp_N, preTrackedKpIndexs, originTrackedKpIndexs, Kps, KpRights, npKeyPointsObsCovariance_array, npKeyPointsPostCovariance_array)


            # Current pose
            R_w_s = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
            R_w_c = R_w_s.dot(R_s_c)
            t_w_c = agent.get_state().sensor_states["left_sensor"].position

            # Set left camera pose
            CurrentFrame.setPose(R_w_c, t_w_c)

            # Compute observation Covariance matrix
            # tmp = CurrentFrame.status_L[11]
            depth_img = np.array(obs["left_sensor_depth"])
            if step == 0:
                # input = np.array([CurrentFrame.Kps[208, 0], CurrentFrame.KpRights[208, 0], CurrentFrame.Kps[208, 1]])
                input  = np.array([198, 186, 373])
                targetPositionW = SparsePipeline.targetWordCoordinate(CurrentFrame, input[0], input[2], depth_img[round(input[2]), round(input[0])])

            #Create init MapPoints

            if step == 0:
                for kp_index in range(CurrentFrame.N):
                    uL = CurrentFrame.Kps[kp_index, 0]
                    v = CurrentFrame.Kps[kp_index, 1]
                    depth_kp = depth_img[round(v), round(uL)]
                    uR = CurrentFrame.KpRights[kp_index, 0]

                    Kp_loc_W_gt = SparsePipeline.targetWordCoordinate(CurrentFrame, uL, v, depth_kp)
                    newMP = MapPoint.MapPoint(kp_index, CurrentFrame, -1.0, Kp_loc_W_gt)  #do not forget to change -1
                    mapPoint_array.append(newMP)


            # # track single point for localization error computation
            # if step == 0:
            #     preTrackedIndex = 104
            #     uL_trackedKp = CurrentFrame.Kps[preTrackedIndex, 0]
            #     v_trackedKp = CurrentFrame.Kps[preTrackedIndex, 1]
            #     depth_trackedKp = depth_img[round(v_trackedKp), round(uL_trackedKp)]
            #     Kp_loc_W_gt = SparsePipeline.targetWordCoordinate(agent, uL_trackedKp, v_trackedKp, depth_trackedKp)
            #
            # curTrackedIndex, depth_gt = SparsePipeline.trackKeyPointIndex(CurrentFrame, preTrackedIndex, depth_img)
            # if curTrackedIndex == -1:
            #     print("End up this tracking")
            #     break


            # ##vis show tracked point
            # leftDraw = leftImgOrg.copy()
            # rightDraw = rightImgOrg.copy()
            # drawTrackedKeyPoint(leftDraw, rightDraw, CurrentFrame.Kps, CurrentFrame.KpRights, curTrackedIndex)
            # stereo_pair = np.concatenate([leftDraw, rightDraw], axis=1)
            #
            # cv2.imshow("stereo_pair", stereo_pair)
            # keystroke = cv2.waitKey(0)
            # if keystroke == ord("q"):
            #     break



            #Compute observation covariance matrix
            depth_img = np.clip(depth_img, 0, 10)
            SparsePipeline.computeObsCovarianceMatrixForKeyPoints(CurrentFrame, depth_img)

            # Update Covariance matrix
            if step == 0:
                LastFrame = CurrentFrame

            SparsePipeline.computePostCovarianceMatrixForKeyPoints(LastFrame, CurrentFrame)


            #trace and localization error
            if step == 0:
                mapPoint_num = len(mapPoint_array)
                kpsTrace_array = np.zeros((mapPoint_num, 200)) #equal to step
                Kpsloc_error_array = np.zeros((mapPoint_num, 200))


                for mp_index in range(mapPoint_num):
                    kpsTrace_array[mp_index][step] = np.trace(CurrentFrame.mPostMatrices_array[mp_index])
                    Kpsloc_error_array[mp_index][step] = SparsePipeline.computeLocErrorForTrackedKeypoint(CurrentFrame, mp_index, mapPoint_array[mp_index].worldPostion_gt)
            else:
                depth_img_t = np.array(obs["left_sensor_depth"])
                for kp_index in range(CurrentFrame.N):
                    kp_original_index = CurrentFrame.originTrackedKpIndexs[kp_index]
                    kpsTrace_array[kp_original_index][step] = np.trace(CurrentFrame.mPostMatrices_array[kp_index])
                    depth_kp = depth_img_t[round(CurrentFrame.Kps[kp_index, 1]), round(CurrentFrame.Kps[kp_index, 0])]
                    if step == 1:
                        mpPosW = SparsePipeline.targetWordCoordinate(CurrentFrame, CurrentFrame.Kps[kp_index, 0], CurrentFrame.Kps[kp_index, 1], depth_kp)
                        mapPoint_array[kp_original_index].worldPostion_gt = mpPosW
                    # if step == 26 and kp_original_index == 723:
                    #     print("HHH")
                    # if kp_original_index == 723:
                    #     uL_gt, uR_gt, v_gt = SparsePipeline.project(CurrentFrame, mapPoint_array[kp_original_index].worldPostion_gt)
                    #     input_t = np.array([CurrentFrame.Kps[kp_index, 0], CurrentFrame.Kps[kp_index, 1]])
                    Kpsloc_error_array[kp_original_index][step] = SparsePipeline.computeLocErrorForTrackedKeypoint(CurrentFrame, kp_index, mapPoint_array[kp_original_index].worldPostion_gt)

                #test
                # leftFatherDraw = leftFather.copy()
                # leftImgDraw = leftImgOrg.copy()
                # print("Enter any key {0}".format(step))
                # drawFeaturePoints(leftFatherDraw, leftImgDraw, FirstFrame, CurrentFrame)
                #
                #
                # stereo_pair = np.concatenate([leftFatherDraw, leftImgDraw], axis=1)
                # # tereo_pair = np.concatenate([leftOld, leftImgOrg], axis=1)w
                # cv2.imshow("stereo_pair", stereo_pair)
                # keystroke = cv2.waitKey(0)
                # if keystroke == ord("q"):
                #     break

            # Select active keypoint for next best view
            # activeKpIndex = SparsePipeline.selectActiveKeyPoint(CurrentFrame)
            #
            #
            # print("Active keypoint index: {0}".format(activeKpIndex))
            # if activeKpIndex == -1:
            #     print("Can not find active point!!!")
            #     break
            # acitiveKp_uL = CurrentFrame.Kps[activeKpIndex, 0]
            # acitiveKp_v = CurrentFrame.Kps[activeKpIndex, 1]
            # acitiveKp_uR = CurrentFrame.KpRights[activeKpIndex, 0]


            # ##VIS: show left-right and active
            # leftDraw = leftImgOrg.copy()
            # rightDraw = rightImgOrg.copy()
            # drawLeftRightFeaturePointsWithActivePoint(leftDraw, rightDraw, CurrentFrame.Kps, CurrentFrame.KpRights, activeKpIndex)
            # stereo_pair = np.concatenate([leftDraw, rightDraw], axis=1)
            #
            # cv2.imshow("stereo_pair", stereo_pair)
            # keystroke = cv2.waitKey(0)
            # if keystroke == ord("q"):
            #     break
            #end select active keypoint

            # +++Control agent to move to NBV position+++
            if step > 0:
                input = TestFun.targetPixelInCurrentCamera(agent, targetPositionW) #delete it after test
            # input = np.array([acitiveKp_uL, acitiveKp_uR, acitiveKp_v])
            # Compute next best position of active keypoint point
            input_withoutOffset = np.array([input[0] - cx, input[1] - cx, input[2] - cy])
            level = depth_img[round(input[2]), round(input[0])]
            print("Active keypoint level: {0}".format(level/10*7))

            #visulaze preleft-curleft
            if len(cur_pt) > 0:
                if step == 0:
                    leftOldDraw = leftImgOrg.copy()
                    pre_pt = cur_pt
                else:
                    leftOldDraw = leftOld.copy()
                leftImgDraw = leftImgOrg.copy()
                drawFeaturePoints(leftOldDraw, leftImgDraw, LastFrame, CurrentFrame)
                # drawFeaturePoints(leftOldDraw, leftImgDraw, LastFrame, CurrentFrame)
                print("Enter any key")
                stereo_pair = np.concatenate([leftOldDraw, leftImgDraw], axis=1)
                # tereo_pair = np.concatenate([leftOld, leftImgOrg], axis=1)w
                cv2.imshow("stereo_pair", stereo_pair)
                keystroke = cv2.waitKey(0)
                if keystroke == ord("q"):
                    break

            # U_prior = LastFrame.mPostMatrices_array[activeKpIndex]

            U_obs = TestFun.computeObsCovariance(agent, depth_img, targetPositionW)

            if step == 0:
                U_prior = U_obs

            U_post, _ = TestFun.objectiveFun(U_prior, U_obs)

            delta = sp.GradientForTarget(input_withoutOffset, level, U_prior, R_w_c)
            print("delta: {0}".format(delta / np.linalg.norm(delta)))

            #Compute next best position for active point in cycolpean coordinate
            targetPositionCy, targetPositionCyNext = SparsePipeline.getNextBestPositionofFeature(input_withoutOffset, delta)
            print("targetPositionCNext in cyclopean coordinate: {0}".format(targetPositionCyNext))

            # compute next best position for camera
            nextBestCameraPostionIncy = SparsePipeline.computeCameraTranslation(agent, targetPositionCy, targetPositionCyNext)

            # compute next best rotation for camera
            R = SparsePipeline.rotateCameraToMidpoint(targetPositionCyNext)  # Rck-1ck
            Rws = quaternion.as_rotation_matrix(agent.get_state().sensor_states["left_sensor"].rotation)
            Rwc = Rws.dot(R_s_c)
            Rwcy = Rwc  # R(tk-1)
            Rwcy_rectify = Rwcy.dot(R.T)

            # update camera position
            SparsePipeline.setAgentPosition(agent, nextBestCameraPostionIncy)
            SparsePipeline.setAgentRotation(agent, Rwcy_rectify)

            # Compute average localization
            leftOld = leftImgOrg.copy()
            if step == 0:
                leftFather = leftImgOrg.copy()
                FirstFrame = copy.deepcopy(CurrentFrame)
            pre_pt = cur_pt
            LastFrame = copy.deepcopy(CurrentFrame)

            #do not forget to delete
            U_prior = U_post

            # preTrackedIndex = curTrackedIndex

        # Plot.plotLocError(loc_error_array, exId)
        # Plot.plotTrace(trace_array, exId)
        print("++++++Start to plot result++++++++++++++++++")
        Plot.plotAllKeyPointsTrace(kpsTrace_array, exId)
        Plot.plotAllKeyPointsLocError(Kpsloc_error_array, exId)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults()
    args = parser.parse_args()
    setupAgentwithSensors(display=args.display)