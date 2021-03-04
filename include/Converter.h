//
// Created by wangweihan on 9/16/20.
//

#ifndef AVP_CONVERTER_H
#define AVP_CONVERTER_H
#include<opencv2/core/core.hpp>
#include<Eigen/Dense>

namespace AVP {

class Converter {
public:
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

    static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);

    static cv::Mat toCvMat(const Eigen::Matrix3d &m);

    static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);

    static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);

    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);

    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);

    static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);

    static std::vector<float> toQuaternion(const cv::Mat &M);

    static Eigen::Matrix<double,4,4> toMatrix4d(const cv::Mat &cvMat4);
};
}
#endif //AVP_CONVERTER_H
