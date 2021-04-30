//
// Created by wangweihan on 8/2/20.
//

#include "System.h"


namespace AVP {
    System::System(const string &strSettingsFile) {
        cout << endl <<
             "Active Viewpoint Planning for 3D Model Acquisition Copyright (C) 2020 Weihan Wang, supervised by Professor Philippos Mordohai."
             << endl;

        mpSparsePipline = new SparsePipline(strSettingsFile);

    }


//    void System::ProcessingStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp) {
//        mpSparsePipline->GrabImageStereo(imLeft, imRight, timestamp);
//    }

    void System::ProcessingStereo(py::array_t<unsigned char>&imLeftpy, py::array_t<unsigned char>&imRightpy, const double &timestamp)
    {
        cv::Mat imLeft = numpy_uint8_3c_to_cv_mat(imLeftpy);
        cv::Mat imRight = numpy_uint8_3c_to_cv_mat(imRightpy);
        mpSparsePipline->GrabImageStereo(imLeft, imRight, timestamp); //Get mcurrentframe after orb feature extraction
    }

    Eigen::Matrix3d System::testEigen(const Eigen::Matrix3d &m1,const Eigen::Matrix3d &m2)
    {
        return m1+m2;
    }

}