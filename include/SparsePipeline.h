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

        Eigen::Matrix4d Track();

        void DelaunayTriangulation(vector<Point<float>> &vPoints, vector<float> &vDepth);

        void WritePTCloud(const vector<Point<float>> &vPoints, const vector<float> &vDepth, const Delaunay<float> &delaunayTriangule);

        void StereoInitialization();

        int SelectActiveKeyPoint();

        Eigen::Vector3d GradientBasedOnNextBestTargetLocation(const int &kpIndex, Eigen::Matrix3d &U_obs, Eigen::Matrix3d &U_prior, Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc);
        
        Eigen::Vector3d Gradient(const Eigen::Vector3d &input, const float &depth, Eigen::Matrix3d &U_prior, Eigen::Matrix3d &Rwc);

        Eigen::Vector3d GradientofCameraTranslation(const Eigen::Vector3d &input, const float &depth, Eigen::Matrix3d &U_prior, Eigen::Matrix3d &Rwc);
        //Update posterior
        pair<Eigen::Vector3d, Eigen::Matrix3d> FuseByKalmanFilter(Eigen::Matrix3d &U_obs, Eigen::Matrix3d &U_prior);

        // Current Frame
        Frame mCurrentFrame;

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
