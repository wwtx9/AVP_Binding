//
// Created by wangweihan on 8/2/20.
//

#ifndef AVP_FRAME_H
#define AVP_FRAME_H
#include<vector>
#include "ORBextractor.h"
#include <opencv2/opencv.hpp>
namespace AVP
{
class Frame
{
public:
    Frame();
    // Constructor for stereo cameras.
   Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    void ExtractORB(int flag, const cv::Mat &im);

    void ComputeStereoMatches();

    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    void UpdatePoseMatrices();
    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints in left image.
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<std::vector<float>> mvKPs;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;


    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    static long unsigned int nNextId;
    long unsigned int mnId;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Camera pose.
    cv::Mat mTcw;

    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mtwc;


};
}
#endif //AVP_FRAME_H
