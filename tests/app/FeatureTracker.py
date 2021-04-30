import numpy as np
import cv2
height = 640
width = 960
lk_params = dict( winSize=(21, 21),
                  maxLevel=5,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def distance(previousKp, currentKp):
    pre_x, pre_y = previousKp.ravel()
    cur_x, cur_y = currentKp.ravel()

    dx = cur_x-pre_x
    dy = cur_y-pre_y

    return np.sqrt(dx*dx + dy*dy)

def inBorder(currentKp):
    u, v = currentKp.ravel()
    u = round(u)
    v = round(v)

    return u >= 1 and u <= width-1 and v >=1 and v <= height-1;



def checkDisparity(Kp_L, Kp_R):
    uL, vL = Kp_L.ravel()
    uR, vR = Kp_R.ravel()

    return abs(vL-vR)


