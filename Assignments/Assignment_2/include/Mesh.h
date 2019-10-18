#ifndef MESH_H
#define MESH_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

using cv::Vec;
std::vector<std::string> split(std::string s, std::string t);
class Mesh
{
public:
    bool loadOBJ(const std::string &filename);
    std::vector<Vec3d> vertices;
    std::vector<Vec3i> faces;
};

#endif