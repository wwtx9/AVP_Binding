//
// Created by wangweihan on 8/19/20.
//

#ifndef AVP_VISUALIZATION_H
#define AVP_VISUALIZATION_H
#include <vector>
#include <iostream>

#include "../src/app/Delaunay.hpp"
namespace AVP
{
class Visualization
{
public:

    Visualization(std::vector<Point<float>> vPoints_);


    std::vector<Point<float>> vPoints;
    std::vector<float> mvDepth;



};
}
#endif //AVP_VISUALIZATION_H
