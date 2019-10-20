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
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include "Mesh.h"

#define PI 3.1415926535897932384626433832795

#define to_rad(angle) angle *(PI / 180.0)

using namespace cv;
using namespace cv::detail;
using cv::Mat;
using cv::Ptr;
using cv::Vec;
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
void find_pose_from_homo(const Mat &H, const Mat &CAM_Intrinsic, Mat &RT);
void render_mesh(Mesh &mesh, Mat img);
void projective_T(Mesh &mesh, Mat projection, Mat translation);
void get_dir_vect_towards_stop_marker(Mat H, Mat projection, Vec3d &dir);

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

    Mat hmask1;
    Mat hmask2;
    Mat H1 = findHomography(kpts_1, kpts_scene_template_1, RANSAC, 3, hmask1, 4000, 0.998);
    Mat H2 = findHomography(kpts_2, kpts_scene_template_2, RANSAC, 3, hmask2, 4000, 0.998);

    Mat src_warped_1, src_warped_2, dst_padded;
    warpPerspectivePadded(template_1, scene, H1.inv(), src_warped_1, dst_padded,
                          WARP_INVERSE_MAP, BORDER_CONSTANT, Scalar());

    warpPerspectivePadded(template_2, scene, H2.inv(), src_warped_2, dst_padded,
                          WARP_INVERSE_MAP, BORDER_CONSTANT, Scalar());

    Mat blended_padded;
    float alpha = 0.4;
    addWeighted(src_warped_1, alpha, dst_padded, (1.0 - alpha), 0.1,
                blended_padded);
    addWeighted(src_warped_2, alpha, blended_padded, (1.0 - alpha), 0.1,
    blended_padded);
  
    // imshow("podh1", dst_padded_1);
    // imshow("podh2", dst_padded_2);

    Mat Cam_Intrinsic = Mat::eye(3, 3, CV_64F);
    Mat RT = Mat::zeros(3, 4, CV_64F);
    Mat RT_stop = Mat::zeros(3, 4, CV_64F);
    Cam_Intrinsic.at<double>(0, 0) = 1097.4228244618459;
    Cam_Intrinsic.at<double>(0, 1) = 0.0;
    Cam_Intrinsic.at<double>(0, 2) = 540.0;
    Cam_Intrinsic.at<double>(1, 0) = 0.0;
    Cam_Intrinsic.at<double>(1, 1) = 1097.4228244618459;
    Cam_Intrinsic.at<double>(1, 2) = 360;
    cout << "CAM INTRINSIC " << Cam_Intrinsic << endl;
    // H1 = -H1;
    find_pose_from_homo(H1, Cam_Intrinsic, RT);
    find_pose_from_homo(H2, Cam_Intrinsic, RT_stop);
    Mat projection = Cam_Intrinsic * RT;

    // Mat reproj_start = (RT.t() * RT).inv() * RT.t();
    // // Mat reproj_stop = (RT_stop.t() * RT_stop).inv() * RT_stop.t();
    // Mat reproj_start;
    // invert(Cam_Intrinsic * RT, reproj_start, DECOMP_SVD);
    // Mat reproj_stop;
    // invert(Cam_Intrinsic * RT_stop, reproj_stop, DECOMP_SVD);

    // Mat dest_vect = reproj_stop * Mat(Vec3d(0.0, 0.0, 1.0));
    // Mat orig_vect = reproj_start * Mat(Vec3d(0.0, 0.0, 1.0));
    // Vec4d dest_v(dest_vect.at<double>(0, 0) / dest_vect.at<double>(0, 3), dest_vect.at<double>(0, 1) / dest_vect.at<double>(0, 3), dest_vect.at<double>(0, 2) / dest_vect.at<double>(0, 3), 1.0);
    // Vec4d orig_v(orig_vect.at<double>(0, 0) / orig_vect.at<double>(0, 3), orig_vect.at<double>(0, 1) / orig_vect.at<double>(0, 3), orig_vect.at<double>(0, 2) / orig_vect.at<double>(0, 3), 1.0);
    // cout
    //     << "STRT" << dest_vect;
    // cout << "STOP" << orig_vect;
    Mat p1 = Cam_Intrinsic * RT_stop * Mat(Vec4d(0.0, 0.0, 0.0, 1.0));
    p1.convertTo(p1, CV_64F);
    int xx = p1.at<double>(0, 0) / p1.at<double>(0, 2);
    int yy = p1.at<double>(0, 1) / p1.at<double>(0, 2);
    cv::circle(blended_padded, Point(xx, yy), 2, Scalar(0, 255, 0), 2);
    // // Mat dest = RT_stop * Mat(Vec4d(0.0, 0.0, 0.0, 1.0));
    // dest.at<double>(0, 0) /= dest.at<double>(0, 2);
    // dest.at<double>(0, 1) /= dest.at<double>(0, 2);
    // dest.at<double>(0, 2) /= dest.at<double>(0, 2);
    // Vec3d dest_v(dest.at<double>(0, 0), dest.at<double>(0, 1), dest.at<double>(0, 2));
    // cout << "DDDDDDDDDDDDDDD" << dest_v << endl;

    // Mat orig = RT * Mat(Vec4d(0.0, 0.0, 0.0, 1.0));
    // orig.at<double>(0, 0) /= orig.at<double>(0, 2);
    // orig.at<double>(0, 1) /= orig.at<double>(0, 2);
    // orig.at<double>(0, 2) /= orig.at<double>(0, 2);
    // Vec3d orig_v(orig.at<double>(0, 0), orig.at<double>(0, 1), orig.at<double>(0, 2));
    // cout << "SSSSSSSSSSSSSSSSS" << orig_v << endl;

    // Vec4d dt = dest_v - orig_v;
    // cout << "DUMMMM" << dt;

    // Vec3d delta_t(dt[0], dt[1], dt[2]);
    // cout << "DELTA" << delta_t;

    // delta_t = delta_t / norm(delta_t);
    Vec3d delta_t(-1.0, 0.0, 0.0);
    Vec3d acc_t(0.0, 0.0, 0.0);
    Mat temp_img;
    blended_padded.copyTo(temp_img); // = blended_padded;
    while (1)
    {
        // cout << "DELTA_T" << delta_t << endl;
        blended_padded.copyTo(temp_img); // = blended_padded;
        Mesh mesh;
        mesh.loadOBJ(meta_parser["mesh"]);
        acc_t += delta_t;
        Eigen::Translation3f t = Eigen::Translation3f(acc_t[0], acc_t[1], acc_t[2]);
        Eigen::Affine3f transform(t);
        Eigen::Matrix4f matrix = transform.matrix();
        Mat translation = Mat::eye(4, 4, CV_64F);
        eigen2cv(matrix, translation);
        projective_T(mesh, projection, translation);
        int c = 0;
        render_mesh(mesh, temp_img);
        imshow("Blended warp, padded crop", temp_img);
        if (c++ == 10000)
            break;
        // temp_img = blended_padded;
        temp_img = Mat::zeros(temp_img.rows, temp_img.cols, CV_8U);
        int keyboard = waitKey(1); // ?
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    // Eigen::AngleAxisd rollAngle(to_rad(0), Eigen::Vector3d::UnitZ());
    // Eigen::AngleAxisd yawAngle(0.0, Eigen::Vector3d::UnitY());
    // Eigen::AngleAxisd pitchAngle(0.0, Eigen::Vector3d::UnitX());
    // Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    // Eigen::Matrix3d rotationMatrix = q.matrix();

    // Eigen::Translation3f t = Eigen::Translation3f(1, 2, 3);
    // Eigen::Affine3f transform(t);
    // Eigen::Matrix4f matrix = transform.matrix();

    // Mat ter;
    // eigen2cv(matrix, ter);
    // cout << "EIGEN MAT IN CV " << ter << endl;
    waitKey(0);
    return 0;
}
void get_dir_vect_towards_stop_marker(Mat H, Mat projection, Vec3d &dir)
{
}
void projective_T(Mesh &mesh, Mat projection, Mat translation)
{
    Mat rot = Mat::zeros(4, 4, CV_64F);
    rot.at<double>(0, 0) = 1.0;
    rot.at<double>(0, 1) = 0.0;
    rot.at<double>(0, 2) = 0.0;
    rot.at<double>(0, 3) = 0.0;
    rot.at<double>(1, 0) = 0.0;
    rot.at<double>(1, 1) = 0.0;
    rot.at<double>(1, 2) = 1.0;
    rot.at<double>(1, 3) = 0.0;
    rot.at<double>(2, 0) = 0.0;
    rot.at<double>(2, 1) = -1.0;
    rot.at<double>(2, 2) = 0.0;
    rot.at<double>(2, 3) = 0.0;
    rot.at<double>(3, 0) = 0.0;
    rot.at<double>(3, 1) = 0.0;
    rot.at<double>(3, 2) = 0.0;
    rot.at<double>(3, 3) = 1.0;
    for (auto &vertex : mesh.vertices)
    {
        vertex = vertex * 1.0;
        Vec4d point_(vertex[0] + (616 / 2), vertex[1] + (416 / 2), vertex[2], 1.0);
        translation.convertTo(translation, CV_64F);
        Mat result = projection * translation * Mat(point_);
        // Mat result = projection * rot * Mat(point_);
        // cout << "SIZECHECK" << endl;
        // Size z = projection.size();
        // cout << " TRANS " << z.height << " " << z.width << endl;
        // cout << "PROJJ " << projection * translation << endl;

        result.at<double>(0, 0) /= result.at<double>(0, 2);
        result.at<double>(0, 1) /= result.at<double>(0, 2);
        result.at<double>(0, 2) /= result.at<double>(0, 2);
        vertex[0] = result.at<double>(0, 0);
        vertex[1] = result.at<double>(0, 1);
        vertex[2] = result.at<double>(0, 2);
    }
}
void render_mesh(Mesh &mesh, Mat img)
{
    for (auto face : mesh.faces)
    {
        Vec3d v1 = mesh.vertices[face[0] - 1];
        Vec3d v2 = mesh.vertices[face[1] - 1];
        Vec3d v3 = mesh.vertices[face[2] - 1];
        cv::line(img, Point(v1[0], v1[1]), Point(v2[0], v2[1]), Scalar(0, 255, 0), 1, LINE_4);
        cv::line(img, Point(v2[0], v2[1]), Point(v3[0], v3[1]), Scalar(0, 255, 0), 1, LINE_4);
        cv::line(img, Point(v3[0], v3[1]), Point(v1[0], v1[1]), Scalar(0, 255, 0), 1, LINE_4);
    }
    // imshow("Run", img);
}
void find_pose_from_homo(const Mat &H, const Mat &CAM_Intrinsic, Mat &RT)
{
    Mat Partial_RT = Mat::zeros(3, 3, CV_64F);

    Partial_RT = CAM_Intrinsic.inv() * H;
    cout << "HOMOHOMO " << H << endl;
    Vec3d g1, g2, g3, t;
    g1[0] = Partial_RT.at<double>(0, 0);
    g1[1] = Partial_RT.at<double>(1, 0);
    g1[2] = Partial_RT.at<double>(2, 0);
    g2[0] = Partial_RT.at<double>(0, 1);
    g2[1] = Partial_RT.at<double>(1, 1);
    g2[2] = Partial_RT.at<double>(2, 1);

    t[0] = Partial_RT.at<double>(0, 2);
    t[1] = Partial_RT.at<double>(1, 2);
    t[2] = Partial_RT.at<double>(2, 2);

    g3 = g1.cross(g2);
    cout << "G1 = " << g1 << endl;
    cout << "G2 = " << g2 << endl;
    cout << "G3 = " << g3 << endl;
    cout << "g1dotg3 = " << g1.dot(g3) << endl;
    cout << "g1dotg2 = " << g1.dot(g2) << endl;
    cout << "g2dotg3 = " << g2.dot(g3) << endl;

    Vec3d r1, r2, t_norm;

    double length_g1g2 = sqrt(norm(g1) * norm(g2));

    r1 = g1 / length_g1g2;
    r2 = g2 / length_g1g2;
    t_norm = t / length_g1g2;

    Vec3d c, p, d;
    c = r1 + r2;
    p = r1.cross(r2);
    d = c.cross(p);

    Vec3d r1dashed = ((c / norm(c)) + (d / norm(d))) * 0.7072135785007072073;
    Vec3d r2dashed = ((c / norm(c)) - (d / norm(d))) * 0.7072135785007072073;
    Vec3d r3dashed = r1dashed.cross(r2dashed);

    cout << "r1' = " << r1dashed << endl;
    cout << "r2' = " << r2dashed << endl;
    cout << "r3' = " << r3dashed << endl;
    cout << "R1dotR3 = " << r1dashed.dot(r3dashed) << endl;
    cout << "R1dotR2 = " << r1dashed.dot(r2dashed) << endl;
    cout << "R2dotR3 = " << r2dashed.dot(r3dashed) << endl;
    RT.at<double>(0, 0) = r1dashed[0];
    RT.at<double>(0, 1) = r2dashed[0];
    RT.at<double>(0, 2) = r3dashed[0];
    RT.at<double>(0, 3) = t_norm[0];
    RT.at<double>(1, 0) = r1dashed[1];
    RT.at<double>(1, 1) = r2dashed[1];
    RT.at<double>(1, 2) = r3dashed[1];
    RT.at<double>(1, 3) = t_norm[1];
    RT.at<double>(2, 0) = r1dashed[2];
    RT.at<double>(2, 1) = r2dashed[2];
    RT.at<double>(2, 2) = r3dashed[2];
    RT.at<double>(2, 3) = t_norm[2];
    cout << "RT " << RT << endl;
    cout << "tnorm " << t_norm;
    cout << "t " << t;
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