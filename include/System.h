//
// Created by wangweihan on 8/2/20.
//

#ifndef AVP_SYSTEM_H
#define AVP_SYSTEM_H
#include<string>
#include<opencv2/core/core.hpp>
#include <iostream>
#include "SparsePipeline.h"
#include "Mat_Wraper.h"
//#include <pybind11/numpy.h>
//#include <pybind11/pybind11.h>

using namespace std;
namespace AVP
{
namespace  py =  pybind11;
class SparsePipline;

class System
{
public:

    System(const string &strSettingsFile);

    void ProcessingStereo(py::array_t<unsigned char>&imLeftpy, py::array_t<unsigned char>&imRightpy, const double &timestamp);
    //void ProcessingStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

    Eigen::Matrix3d testEigen(const Eigen::Matrix3d &m1,const Eigen::Matrix3d &m2);


    SparsePipline* mpSparsePipline;

};
}
#endif //AVP_SYSTEM_H



