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

void removeBackground(Mat &input, Mat &background);
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
    VideoCapture capture(0);
    // VideoCapture capture(video_file);
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2(0, 50);
    int fps = capture.get(cv::CAP_PROP_FPS);
    int count = 1;
    string name = meta_parser["dir_name"];
    string iclassname = name + "_";
    // Mat background = imread("background.jpeg", 0);
    while (true)
    {
        Mat current_frame, colfr;
        capture >> current_frame;
        count++;

        // cvtColor(current_frame, current_frame, cv::COLOR_BGR2GRAY);
        // current_frame.copyTo(colfr);
        // blur(current_frame, current_frame, Size(10, 10));
        // Canny(current_frame, current_frame, 100, 255, 3);
        // blur(current_frame, current_frame, Size(8, 8));

        pBackSub->apply(current_frame, current_frame);
        // current_frame.copyTo(colfr);
        // cvtColor(current_frame, current_frame, COLOR_BGR2HSV);
        // removeBackground(current_frame, background);
        // Detect the object based on HSV Range Values
        // inRange(current_frame, Scalar(0, 58, 50), Scalar(30, 255, 255), current_frame);
        // Canny(current_frame, current_frame, 100, 255, 3);
        imshow("BW Vid", current_frame);
        // bitwise_xor(colfr, current_frame, colfr, noArray());
        // threshold(c  olfr, colfr, 240, 255, THRESH_BINARY);
        // imshow("Col Vid", current_frame);
        // imshow("test Vid", colfr);
        // imwrite("save" + to_string(count) + ".jpeg", current_frame);
        // return 0;

        // resize(current_frame, current_frame, Size(50, 50), cv::INTER_AREA);
        // resize(colfr, colfr, Size(50, 50), cv::INTER_AREA);
        // vector<int> compression_params;
        // compression_params.push_back(IMWRITE_JPEG_QUALITY);
        // compression_params.push_back(100);
        // imwrite(iclassname + to_string(count) + ".jpeg", current_frame, compression_params);
        // imshow("Render Vid", current_frame);
        // imshow("Render Vid", colfr);

        int keyboard = waitKey(1); // ?
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    cout << "Total frames : " << count << endl;
    return 0;
}
void removeBackground(Mat &input, Mat &background)
{
    int thresholdOffset = 30;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            uchar framePixel = input.at<uchar>(i, j);
            uchar bgPixel = background.at<uchar>(i, j);

            if (
                framePixel >= bgPixel - thresholdOffset &&
                framePixel <= bgPixel + thresholdOffset)
                input.at<uchar>(i, j) = 0;
            else
                input.at<uchar>(i, j) = 255;
        }
    }
}
