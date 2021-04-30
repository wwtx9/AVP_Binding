//
// Created by wangweihan on 8/2/20.
//

#ifndef AVP_FRAME_H
#define AVP_FRAME_H
#include<vector>
#include<Eigen/Dense>
#include "ORBextractor.h"
#include "Converter.h"
#include <opencv2/opencv.hpp>
namespace AVP
{

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64


struct KeyPoint
{
    float x;
    float y;
    float size;
    float angle;
    float response;
    int octave;
    int class_id = -1;
};

class Frame
{
public:
    Frame();
    Frame(const Frame &frame);
    
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
    // Set the camera pose.
    void SetPoseFromHabitat(Eigen::Matrix3d Rwc, Eigen::Vector3d twc);
    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;
    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;

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
    // std::vector<std::vector<float>> mvKPs;
    std::vector<KeyPoint> mvKPs;
    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    //store observation covariance matrix of each keypoint
    // std::vector<Eigen::Matrix3d> mvKeyPointsObsCovariance;
    // //store posterior covariance matrix of each keypoint
    // std::vector<Eigen::Matrix3d> mvKeyPointsPostCovariance;

    std::vector<int> mvmatchedNewKeypointsIndex;
    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;
    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    float mnMinX;
    float mnMaxX;
    float mnMinY;
    float mnMaxY;


    float mfGridElementWidthInv;
    float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw;

    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mtwc;

private:
    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();


};
}
#endif //AVP_FRAME_H
