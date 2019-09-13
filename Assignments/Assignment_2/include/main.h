#include <iostream>
#include <sstream>
#include <fstream>
#include "json.hpp"
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp> //Thanks to Alessandro
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp> // drawing the COG
#include <opencv2/calib3d.hpp>

#include "f_utils.h"

using namespace cv;
using cv::Mat;
using cv::Ptr;
using cv::xfeatures2d::SIFT;
using namespace std;
using json = nlohmann::json;

// prototypes
void get_keypoints(Mat &input, vector<KeyPoint> &kpts, Mat &desc);
void show_keypoints(Mat &input, Mat &output, vector<KeyPoint> &kpts);
Point2f compute_COG(vector<KeyPoint> &kpts);
void populate_point2f_keypoint_vector(std::vector<Point2f> &kpts_as_point2f, vector<KeyPoint> &kpts);
inline void match(Mat &desc1, Mat &desc2, vector<DMatch> &matches);
void sample_down(vector<Mat> &all_images);
void get_keypoints_and_descriptors_for_all_imgs(vector<Mat> &input, vector<vector<KeyPoint>> &kpts, vector<Mat> &desc);

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;

void get_keypoints(Mat &input, vector<KeyPoint> &kpts, Mat &desc)
{
    Ptr<Feature2D> sift = SIFT::create();
    sift->detectAndCompute(input, Mat(), kpts, desc);
}
void get_keypoints_and_descriptors_for_all_imgs(vector<Mat> &all_images, vector<vector<KeyPoint>> &keypoint_all_img, vector<Mat> &descriptors_all_img)
{
    for (auto img : all_images)
    {
        vector<KeyPoint> kpt;
        Mat des;
        get_keypoints(img, kpt, des);
        keypoint_all_img.push_back(kpt);
        descriptors_all_img.push_back(des);
    }
}
void show_keypoints(Mat &input, Mat &output, vector<KeyPoint> &kpts)
{
    drawKeypoints(input, kpts, output);
}

Point2f compute_COG(vector<KeyPoint> &kpts)
{
    Point2f results;
    for (auto kpt : kpts)
    {
        results.x += kpt.pt.x;
        results.y += kpt.pt.y;
    }
    results.x /= kpts.size();
    results.y /= kpts.size();
    return results;
}

void populate_point2f_keypoint_vector(std::vector<Point2f> &kpts_as_point2f, vector<KeyPoint> &kpts)
{
    for (auto keypoint : kpts)
    {
        kpts_as_point2f.push_back(keypoint.pt);
    }
}
inline void match(Mat &desc1, Mat &desc2, vector<DMatch> &matches)
{
    matches.clear();
    BFMatcher desc_matcher(cv::NORM_L2, true);
    vector<vector<DMatch>> vmatches;
    desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
    for (int i = 0; i < static_cast<int>(vmatches.size()); ++i)
    {
        if (!vmatches[i].size())
        {
            continue;
        }
        matches.push_back(vmatches[i][0]);
    }

    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance)
    {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize)
    {
        matches.pop_back();
    }
}

void sample_down(vector<Mat> &all_images)
{
    for (auto &img : all_images)
    {
        resize(img, img, Size(img.size().width / 6, img.size().height / 6), cv::INTER_AREA);
    }
}