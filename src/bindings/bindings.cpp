//
// Created by wangweihan on 2/3/21.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "System.h"
#include "SparsePipeline.h"
#include "Frame.h"

namespace AVP{
    //Binding
    namespace py = pybind11;

    PYBIND11_MODULE(AVP_Binding, m) {
        py::class_<SparsePipline>(m, "SparsePipline")
                .def(py::init<const string&>())
                .def("GrabImageStereo", &SparsePipline::GrabImageStereo)
                .def("DelaunayTriangulation", &SparsePipline::DelaunayTriangulation)
                .def("StereoInitialization", &SparsePipline::StereoInitialization)
                //.def("ComputeObsCovarianceMatrix", &SparsePipline::ComputeObsCovarianceMatrix) //Obs cov matrix
                //.def("UpdateCovarianceMatrix", &SparsePipline::UpdateCovarianceMatrix) //update cov matrix
                //.def("SelectActiveKeyPoint", &SparsePipline::SelectActiveKeyPoint)
                .def("GradientForTarget", &SparsePipline::GradientForTarget)
                .def("GradientForActiveKeypoint", &SparsePipline::GradientForActiveKeypoint)
                .def("TrackKeyPoints", &SparsePipline::TrackKeyPoints)
                .def("WritePTCloud", &SparsePipline::WritePTCloud)
                .def_readwrite("mCurrentFrame", &SparsePipline::mCurrentFrame);

        //Frame
        py::class_<Frame>(m, "Frame", py::dynamic_attr())    
                .def(py::init<>())
                .def(py::init<const Frame &>())
                .def("SetPoseFromHabitat", &Frame::SetPoseFromHabitat) //instead SLAM 
                //.def(py::init<const cv::Mat &, const cv::Mat &, const double &, ORBextractor* , ORBextractor* , cv::Mat &, cv::Mat &, const float &, const float &>())
                .def_readonly("mvKPs", &Frame::mvKPs)
                //.def_readwrite("mnFrameId", &Frame::mnFrameId)
                .def_readonly("mvuRight", &Frame::mvuRight)
                //.def_readwrite("mvKeyPointsPostCovariance", &Frame::mvKeyPointsPostCovariance)
                //.def_readwrite("mvKeyPointsObsCovariance", &Frame::mvKeyPointsObsCovariance)
                .def_readonly("mvmatchedNewKeypointsIndex", &Frame::mvmatchedNewKeypointsIndex);

        py::class_<KeyPoint>(m, "KeyPoint")
                .def(py::init<>())
                .def_readonly("x", &KeyPoint::x)
                .def_readonly("y", &KeyPoint::y)
                .def_readonly("size", &KeyPoint::size)
                .def_readonly("angle", &KeyPoint::angle)
                .def_readonly("response", &KeyPoint::response)
                .def_readonly("octave", &KeyPoint::octave)
                .def_readonly("class_id", &KeyPoint::class_id);


        py::class_<System>(m, "System")
                .def(py::init<const string &>())
                .def("testEigen", &System::testEigen)
                .def("ProcessingStereo", &System::ProcessingStereo, py::arg("imLeft"), py::arg("imRight"), py::arg("timestamp"))
                .def_readonly("mpSparsePipline", &System::mpSparsePipline);
    }
}



