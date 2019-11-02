#include <iostream>
#include <sstream>
#include <fstream>
#include "json.hpp"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp> //Thanks to Alessandro
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp> // drawing the COG
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include "Mesh.h"

using namespace cv;
using namespace cv::detail;
using cv::Mat;
using cv::Ptr;
using cv::Vec;
using cv::xfeatures2d::SIFT;
using namespace std;
using json = nlohmann::json;

int main(int argc, const char *argv[])
{
#ifdef _MSC_VER
    std::ifstream ifile("C:/Projects/Acads/COL780/Assignments/Assignment_2/input/meta.json");
#else
    std::ifstream ifile("Assignments/Assignment_2/input/meta.json");
#endif
    json meta_parser;
    ifile >> meta_parser;
    const string video_file = meta_parser["data"][3];
    VideoCapture capture(video_file);
    while (true)
    {
        Mat current_frame;
        capture >> current_frame;
        imshow("Render Vid", current_frame);
        int keyboard = waitKey(1); // ?
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
}
