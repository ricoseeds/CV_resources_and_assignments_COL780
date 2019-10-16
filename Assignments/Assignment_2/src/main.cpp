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
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/util.hpp>

using namespace cv;
using namespace cv::detail;
using cv::Mat;
using cv::Ptr;
using cv::xfeatures2d::SIFT;
using namespace std;
using json = nlohmann::json;

// #define DEBUG_MATS
#define DEBUG_MATCHES

// prototypes
void get_keypoints(Mat &input, vector<KeyPoint> &kpts, Mat &desc);
void show_keypoints(Mat &input, Mat &output, vector<KeyPoint> &kpts);
void populate_point2f_keypoint_vector(std::vector<Point2f> &kpts_as_point2f, vector<KeyPoint> &kpts);
inline void match(Mat &desc1, Mat &desc2, vector<DMatch> &matches);
void warpPerspectivePadded(const Mat &src, const Mat &dst, const Mat &M, Mat &src_warped, Mat &dst_padded, int flags, int borderMode, const Scalar &borderValue);

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 100;

int main(int argc, const char *argv[])
{
#ifdef _MSC_VER
    std::ifstream ifile("C:/Projects/Acads/COL780/Assignments/Assignment_2/input/meta.json");
#else
    std::ifstream ifile("Assignments/Assignment_2/input/meta.json");
#endif

    json meta_parser;
    ifile >> meta_parser;

    vector<KeyPoint> kpts_image_1;
    vector<KeyPoint> kpts_image_2;
    vector<KeyPoint> kpts_image_scene;

    Mat desc_1;
    Mat desc_2;
    Mat desc_scene;
    Mat template_1 = imread(meta_parser["data"][0], IMREAD_COLOR); //Load as grayscale
    Mat template_2 = imread(meta_parser["data"][1], IMREAD_COLOR); //Load as grayscale

    get_keypoints(template_1, kpts_image_1, desc_1);
    get_keypoints(template_2, kpts_image_2, desc_2);

    Mat scene = imread(meta_parser["data"][2], IMREAD_COLOR); //Load as grayscale
    get_keypoints(scene, kpts_image_scene, desc_scene);

    std::vector<Point2f> kpts_1;
    std::vector<Point2f> kpts_2;
    std::vector<Point2f> kpts_scene_template_1;
    std::vector<Point2f> kpts_scene_template_2;

    vector<DMatch> matches_template_1;
    vector<DMatch> matches_template_2;
    match(desc_1, desc_scene, matches_template_1);
    match(desc_2, desc_scene, matches_template_2);

#ifdef DEBUG_MATCHES
    Mat img_matches;
    drawMatches(template_2, kpts_image_2, scene, kpts_image_scene, matches_template_2, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("matched_image", img_matches);
#endif

    if (static_cast<int>(matches_template_1.size()) < 3 && static_cast<int>(matches_template_2.size()) < 3)
    {
        cout << "Not enough correspondence";
        return 0;
    }
    for (int i = 0; i < static_cast<int>(matches_template_1.size()); ++i)
    {
        kpts_1.push_back(kpts_image_1[matches_template_1[i].queryIdx].pt);
        kpts_scene_template_1.push_back(kpts_image_scene[matches_template_1[i].trainIdx].pt);
    }
    for (int i = 0; i < static_cast<int>(matches_template_2.size()); ++i)
    {
        kpts_2.push_back(kpts_image_2[matches_template_2[i].queryIdx].pt);
        kpts_scene_template_2.push_back(kpts_image_scene[matches_template_2[i].trainIdx].pt);
    }

    //Matched images will have least normalized distance.
    // float normalized_distance = std::accumulate(keypoints_distance.begin(), keypoints_distance.end(), 0) / keypoints_distance.size();

    // We must find a way to keep cache this distance, then attach images based on this distance.
    // std::cout << "normalized_distance: " << normalized_distance << std::endl;

    Mat hmask1;
    Mat hmask2;
    Mat H1 = findHomography(kpts_1, kpts_scene_template_1, RANSAC, 3, hmask1, 4000, 0.998);
    Mat H2 = findHomography(kpts_2, kpts_scene_template_2, RANSAC, 3, hmask2, 4000, 0.998);
    // cout << "HOMO mask : " << hmask;
    // cout << "Keypoint 1 size = " << kpts_1.size() << " Keypoint_2_size = " << kpts_2.size() << endl;
    // cout << "Homography matrix  : " << H << endl;

    // cv::Mat result;
    // warpPerspective(input_2, result, H.inv(), cv::Size(input_1.cols + input_2.cols, input_2.rows), INTER_CUBIC);
    // imshow("Result_warped", result);
    double scale_factor = 2.0; // Our tuning knob for image size for the final stitched image
    Mat src_warped, dst_padded;
    Mat scl = Mat::eye(3, 3, CV_64F);
    scl = scl * scale_factor;
    scl.at<double>(2, 2) = 1;
    H2 = scl.inv() * H2 * scl;
    resize(template_1, template_1, Size(template_1.size().width / scale_factor, template_1.size().height / scale_factor), cv::INTER_AREA);
    resize(template_2, template_2, Size(template_2.size().width / scale_factor, template_2.size().height / scale_factor), cv::INTER_AREA);
    resize(scene, scene, Size(scene.size().width / scale_factor, scene.size().height / scale_factor), cv::INTER_AREA);

    // warpPerspectivePadded(template_1, scene, H1.inv(), src_warped, dst_padded,
    //                       WARP_INVERSE_MAP, BORDER_CONSTANT, Scalar());

    warpPerspectivePadded(template_2, scene, H2.inv(), src_warped, dst_padded,
                          WARP_INVERSE_MAP, BORDER_CONSTANT, Scalar());

    Mat blended_padded;
    float alpha = 0.4;
    addWeighted(src_warped, alpha, dst_padded, (1.0 - alpha), 0.1,
                blended_padded);
    imshow("Blended warp, padded crop", blended_padded);

    //BlendLaplacian(input_1, result);

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
void warpPerspectivePadded(
    const Mat &src, const Mat &dst, const Mat &M,         // input matrices
    Mat &src_warped, Mat &dst_padded,                     // output matrices
    int flags, int borderMode, const Scalar &borderValue) // OpenCV params
{

#ifdef DEBUG_MATS
    cout << __FUNCTION__ << " " << __LINE__ << " Homography Mat M " << endl
         << M << endl;
#endif

    Mat transf = M / M.at<double>(2, 2); // ensure a legal homography

#ifdef DEBUG_MATS
    cout << __FUNCTION__ << " " << __LINE__ << " transf " << endl
         << transf << endl;
#endif

    if (flags == WARP_INVERSE_MAP ||
        flags == INTER_LINEAR + WARP_INVERSE_MAP ||
        flags == INTER_NEAREST + WARP_INVERSE_MAP)
    {
        invert(transf, transf);
        flags -= WARP_INVERSE_MAP;
    }

#ifdef DEBUG_MATS
    cout << __FUNCTION__ << " " << __LINE__ << " transf " << endl
         << transf << endl;
#endif
    // it is enough to find where the corners of the image go to find
    // the padding bounds; points in clockwise order from origin
    // transforming the source

    // SR>> Acutally you are taking anti-clockwise order
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

#ifdef DEBUG_MATS
    cout << __FUNCTION__ << " " << __LINE__ << "transl_transf" << endl
         << transl_transf << endl;
#endif

    transf = transl_transf * transf;
    transf /= transf.at<float>(2, 2);

#ifdef DEBUG_MATS
    cout << __FUNCTION__ << " " << __LINE__ << "transl_transf" << endl
         << transl_transf << endl;
#endif

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

    imshow("dst", dst);
    copyMakeBorder(dst, dst_padded, pad_top, pad_bot, pad_left, pad_right, borderMode, borderValue);
    //imshow("dst_padded", dst_padded);
    imwrite("C:/Projects/Acads/out/dst_padded.jpg", dst_padded);

    // transform src into larger window
    int dst_pad_h = dst_padded.rows;
    int dst_pad_w = dst_padded.cols;
    imshow("src", src);
    warpPerspective(src, src_warped, transf, Size(dst_pad_w, dst_pad_h),
                    flags, borderMode, borderValue);

    //imshow("src_warped", src_warped);
    imwrite("C:/Projects/Acads/out/src_warped.jpg", src_warped);
}