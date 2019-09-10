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
inline void match(string type, Mat &desc1, Mat &desc2, vector<DMatch> &matches);

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;

int main(int argc, const char *argv[])
{
#ifdef _MSC_VER
    std::ifstream ifile("C:/Projects/Acads/COL780/Assignments/Assignment_2/input/meta.json");
#else
    std::ifstream ifile("Assignments/Assignment_2/input/meta.json");
#endif
    json meta_parser;
    ifile >> meta_parser;
    //
    vector<KeyPoint> kpts_image_1;
    vector<KeyPoint> kpts_image_2;
    Mat desc_1;
    Mat desc_2;
    Mat input_1 = imread(meta_parser["data"][0], 0); //Load as grayscale
    Mat input_2 = imread(meta_parser["data"][1], 0); //Load as grayscale
    Mat output;                                      // output of sift
    get_keypoints(input_1, kpts_image_1, desc_1);
    get_keypoints(input_2, kpts_image_2, desc_2);
    imshow("img", input_1);

    // Add results to image and save.
    // show_keypoints(input_1, output, kpts);

    std::vector<Point2f> kpts_1;
    std::vector<Point2f> kpts_2;

    vector<DMatch> matches;
    match("knn", desc_1, desc_2, matches);
    vector<char> match_mask(matches.size(), 1);
    if (static_cast<int>(match_mask.size()) < 3)
    {
        cout << "Not enough correspondence";
        return 0;
    }
    for (int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        kpts_1.push_back(kpts_image_1[matches[i].queryIdx].pt);
        kpts_2.push_back(kpts_image_2[matches[i].trainIdx].pt);
    }
    Mat H = cv::findHomography(kpts_1,
                               kpts_2,
                               0,
                               3,
                               noArray(),
                               2000,
                               0.995);
    cout << "Keypoint 1 size = " << kpts_1.size() << " Keypoint_2_size = " << kpts_2.size() << endl;
    cout << "Homography matrix  : " << H << endl;

    waitKey(0);
    return 0;
}

void get_keypoints(Mat &input, vector<KeyPoint> &kpts, Mat &desc)
{
    Ptr<Feature2D> sift = SIFT::create();
    sift->detectAndCompute(input, Mat(), kpts, desc);
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
inline void match(string type, Mat &desc1, Mat &desc2, vector<DMatch> &matches)
{
    matches.clear();
    if (type == "bf")
    {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        desc_matcher.match(desc1, desc2, matches, Mat());
    }
    if (type == "knn")
    {
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