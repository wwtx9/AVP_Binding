//
// Created by wangweihan on 2/6/21.
//

#ifndef AVP_BINDING_MAT_WRAPER_H
#define AVP_BINDING_MAT_WRAPER_H
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>


namespace AVP
{
    namespace py = pybind11;

    cv::Mat numpy_uint8_1c_to_cv_mat(py::array_t<unsigned char>& input);

    cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input);

    py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(cv::Mat & input);

    py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat & input);

}

#endif //AVP_BINDING_MAT_WRAPER_H
