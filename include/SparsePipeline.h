//
// Created by wangweihan on 8/2/20.
//

#ifndef AVP_TRACKING_H
#define AVP_TRACKING_H
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <fstream>
#include<Eigen/Dense>

#include "System.h"
#include"ORBextractor.h"
#include"Frame.h"
#include "../src/app/Delaunay.hpp"

using namespace std;
namespace AVP
{
    class System;

    class SparsePipline
    {

    public:
        SparsePipline(const string &strSettingPath);

        void GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);

        // Computes the Hamming distance between two ORB descriptors
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        Eigen::Matrix4d Track();

        void DelaunayTriangulation(vector<Point<float>> &vPoints, vector<float> &vDepth);

        void WritePTCloud(const vector<Point<float>> &vPoints, const vector<float> &vDepth, const Delaunay<float> &delaunayTriangule);

        void StereoInitialization();

        //Gradient of objective function respect to target position using cyclopean coordinate
        Eigen::Vector3d GradientForTarget(const Eigen::Vector3d &input, const float &depth, const Eigen::Matrix3d &U_prior, const Eigen::Matrix3d &Rwc);

        //Gradient of objective function respect to target position using left camera coordinate
        Eigen::Vector3d Gradient(const Eigen::Vector3d &input, const float &depth, Eigen::Matrix3d &U_prior, Eigen::Matrix3d &Rwc);

        //Gradient of objective function respect to active keypoint position using cyclopean coordinate
        Eigen::Vector3d GradientForActiveKeypoint(const Eigen::Vector3d &input, const float &level, const Eigen::Matrix3d &U_prior, const Eigen::Matrix3d &Rwc);

        Eigen::Matrix3d MakeJacobian(const Eigen::Vector3d &input);

        Eigen::Matrix3d MakeQ(const int level);
        
        std::vector<int> TrackKeyPoints(Frame &CurrentFrame, const Frame &LastFrame, const float th = 15, const bool mbCheckOrientation = true);

        void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

        //Compute observation covariance matrix for each visible keypoints
        //vector<Eigen::Matrix3d> ComputeObsCovarianceMatrix(const Eigen::Matrix3d &Rwc); //remove const int restrictedNum

        //Update covariance matrix by kalman filter
        //void UpdateCovarianceMatrix(const Frame &LastFrame, const Eigen::Matrix3d &Rwc);

        //int SelectActiveKeyPoint();

        //Update posterior
        pair<Eigen::Vector3d, Eigen::Matrix3d> FuseByKalmanFilter(Eigen::Matrix3d &U_obs, Eigen::Matrix3d &U_prior);

        // Current Frame
        Frame mCurrentFrame;
        Frame mLastFrame;
        
        enum eTrackingState{
            SYSTEM_NOT_READY=-1,
            NOT_INITIALIZED=0,
            OK=1,
            LOST=2
        };

        eTrackingState mState;


    
        cv::Mat mK;
        cv::Mat mDistCoef;
        float mbf;
        float mb;

        Eigen::Matrix3d mRwc;
        Eigen::Vector3d mtwc;

        //Color order (true RGB, false BGR, ignored if grayscale)
        bool mbRGB;
        //ORB
        ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
        float mThDepth;

        

    };

}
#endif //AVP_TRACKING_H
