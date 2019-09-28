#include <iostream>
#include <sstream>
#include <fstream>
#include "json.hpp"
#include <vector>
#include <map>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp> //Thanks to Alessandro
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp> // drawing the COG
#include <opencv2/calib3d.hpp>

#include "f_utils.h"
#include "bfs.h"

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
void warpPerspectivePadded(
    const Mat &src, const Mat &dst, const Mat &M, // input matrices
    Mat &src_warped, Mat &dst_padded,             // output matrices
    int flags, int borderMode, const Scalar &borderValue);
const double kDistanceCoef = 4.0;
int kMaxMatchingSize = 0;

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
void warpPerspectivePadded(
    const Mat &src, const Mat &dst, const Mat &M,         // input matrices
    Mat &src_warped, Mat &dst_padded,                     // output matrices
    int flags, int borderMode, const Scalar &borderValue) // OpenCV params
{

    Mat transf = M / M.at<double>(2, 2); // ensure a legal homography
    if (flags == WARP_INVERSE_MAP ||
        flags == INTER_LINEAR + WARP_INVERSE_MAP ||
        flags == INTER_NEAREST + WARP_INVERSE_MAP)
    {
        invert(transf, transf);
        flags -= WARP_INVERSE_MAP;
    }

    // it is enough to find where the corners of the image go to find
    // the padding bounds; points in clockwise order from origin
    // transforming the source
    int dst_h = dst.rows;
    int dst_w = dst.cols;
    Point2f fixed_1 = Point2f(0, 0);
    Point2f fixed_2 = Point2f(dst_w, 0);
    Point2f fixed_3 = Point2f(dst_w, dst_h);
    Point2f fixed_4 = Point2f(0, dst_h);
    cout << "original_DIM" << endl;
    cout << "fixed_1 : " << fixed_1 << endl;
    cout << "fixed_2 : " << fixed_2 << endl;
    cout << "fixed_3 : " << fixed_3 << endl;
    cout << "fixed_4 : " << fixed_4 << endl;

    int src_h = src.rows;
    int src_w = src.cols;
    Vec<Point2f, 4> init_pts, transf_pts;
    cout << "dest BEFORE transform" << endl;
    init_pts[0] = Point2f(0, 0);
    init_pts[1] = Point2f(src_w, 0);
    init_pts[2] = Point2f(src_w, src_h);
    init_pts[3] = Point2f(0, src_h);
    cout << "init_pts[0] : " << init_pts[0] << endl;
    cout << "init_pts[1] : " << init_pts[1] << endl;
    cout << "init_pts[2] : " << init_pts[2] << endl;
    cout << "init_pts[3] : " << init_pts[3] << endl;
    // perspective transform
    perspectiveTransform(init_pts, transf_pts, transf);
    cout << "MAT << " << transf << endl;
    cout << "dest AFTER transform" << endl;
    cout << "init_pts[0] : " << transf_pts[0] << endl;
    cout << "init_pts[1] : " << transf_pts[1] << endl;
    cout << "init_pts[2] : " << transf_pts[2] << endl;
    cout << "init_pts[3] : " << transf_pts[3] << endl;

    // find min and max points
    int min_x, min_y, max_x, max_y;
    min_x = floor(min(
        min(transf_pts[0].x, transf_pts[1].x),
        min(transf_pts[2].x, transf_pts[3].x)));
    min_y = floor(min(
        min(transf_pts[0].y, transf_pts[1].y),
        min(transf_pts[2].y, transf_pts[3].y)));
    max_x = ceil(max(
        max(transf_pts[0].x, transf_pts[1].x),
        max(transf_pts[2].x, transf_pts[3].x)));
    max_y = ceil(max(
        max(transf_pts[0].y, transf_pts[1].y),
        max(transf_pts[2].y, transf_pts[3].y)));

    // add translation to transformation matrix to shift to positive values
    int anchor_x = 0, anchor_y = 0;
    Mat transl_transf = Mat::eye(3, 3, CV_32F);
    if (min_x < 0)
    {
        anchor_x = -min_x;
        transl_transf.at<float>(0, 2) += anchor_x;
    }
    if (min_y < 0)

    {
        anchor_y = -min_y;
        transl_transf.at<float>(1, 2) += anchor_y;
    }

    transl_transf.convertTo(transl_transf, CV_64F);
    transf = transl_transf * transf;
    transf /= transf.at<float>(2, 2);

    // create padded destination image
    // int dst_h = dst.rows;
    // int dst_w = dst.cols;
    int pad_top = anchor_y;
    int pad_bot = max(max_y, dst_h) - dst_h;
    int pad_left = anchor_x;
    int pad_right = max(max_x, dst_w) - dst_w;

    cout << "pad_top : " << pad_top << endl;
    cout << "pad_bot : " << pad_bot << endl;
    cout << "pad_left : " << pad_left << endl;
    cout << "pad_right : " << pad_right << endl;
    copyMakeBorder(dst, dst_padded, pad_top, pad_bot, pad_left, pad_right, borderMode, borderValue);
    // imshow("indst", dst_padded);

    // transform src into larger window
    int dst_pad_h = dst_padded.rows;
    int dst_pad_w = dst_padded.cols;
    warpPerspective(src, src_warped, transf, Size(dst_pad_w, dst_pad_h),
                    flags, borderMode, borderValue);
}