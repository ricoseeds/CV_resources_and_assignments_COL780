#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/stat.h> 
#include <sys/types.h> 
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
    const string video_file = meta_parser["data"][0];
    VideoCapture capture(video_file);
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();
    int fps = capture.get(cv::CAP_PROP_FPS);
    int count = 1;
    string name = meta_parser["dir_name"];
    string iclassname = name + "_";
    while (true)
    {
        Mat current_frame;
        capture >> current_frame;
        count ++;
        resize(current_frame, current_frame, Size(50, 50), cv::INTER_AREA);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_JPEG_QUALITY);
        compression_params.push_back(100);
        imwrite(iclassname + to_string(count) + ".jpeg", current_frame, compression_params);
        // imshow("Render Vid", current_frame);
        // cout << "Total frames : " << count << endl;

        int keyboard = waitKey(1); // ?
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    cout << "Total frames : " << count << endl;
    return 0;
}
