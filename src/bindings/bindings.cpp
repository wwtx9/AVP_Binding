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
                .def("SelectActiveKeyPoint", &SparsePipline::SelectActiveKeyPoint)
                .def("GradientForTarget", &SparsePipline::GradientForTarget)
                .def("Gradient", &SparsePipline::Gradient)
                .def("GradientofCameraTranslation", &SparsePipline::GradientofCameraTranslation)
                .def("WritePTCloud", &SparsePipline::WritePTCloud)
                .def_readwrite("mCurrentFrame", &SparsePipline::mCurrentFrame);


        //Frame
        py::class_<Frame>(m, "Frame")
                .def(py::init<>())
                .def(py::init<const cv::Mat &, const cv::Mat &, const double &, ORBextractor* , ORBextractor* , cv::Mat &, cv::Mat &, const float &, const float &>())
                .def("GetPose", &Frame::GetPose)
                .def("GetPoseInverse", &Frame::GetPoseInverse)
                .def("GetRotation", &Frame::GetRotation)
                .def("GetTranslation", &Frame::GetTranslation)
                .def("ExtractORB", &Frame::ExtractORB)
                .def("ComputeStereoMatches", &Frame::ComputeStereoMatches)
                .def("DescriptorDistance", &Frame::DescriptorDistance)
                .def("SetPose", &Frame::SetPose)
                .def("UpdatePoseMatrices", &Frame::UpdatePoseMatrices)
                .def_readwrite("mvKPs", &Frame::mvKPs)
                .def_readwrite("mvuRight", &Frame::mvuRight);

        py::class_<System>(m, "System")
                .def(py::init<const string &>())
                .def("testEigen", &System::testEigen)
                .def("ProcessingStereo", &System::ProcessingStereo, py::arg("imLeft"), py::arg("imRight"), py::arg("timestamp"))
                .def_readwrite("mpSparsePipline", &System::mpSparsePipline);
    }
}
