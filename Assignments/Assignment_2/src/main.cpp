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
inline void match(Mat &desc1, Mat &desc2, vector<DMatch> &matches);
int BlendLaplacian(Mat l8u, Mat r8u);
const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 20;


// Reference
// https://www.morethantechnical.com/2011/11/13/just-a-simple-laplacian-pyramid-blender-using-opencv-wcode/
class LaplacianBlending {
private:
	Mat_<Vec3f> left;
	Mat_<Vec3f> right;
	Mat_<float> blendMask;

	vector<Mat_<Vec3f> > leftLapPyr, rightLapPyr, resultLapPyr;
	Mat leftSmallestLevel, rightSmallestLevel, resultSmallestLevel;
	vector<Mat_<Vec3f> > maskGaussianPyramid; //masks are 3-channels for easier multiplication with RGB

	int levels;


	void buildPyramids() {
		buildLaplacianPyramid(left, leftLapPyr, leftSmallestLevel);
		buildLaplacianPyramid(right, rightLapPyr, rightSmallestLevel);
		buildGaussianPyramid();
	}

	void buildGaussianPyramid() {
		assert(leftLapPyr.size()>0);

		maskGaussianPyramid.clear();
		Mat currentImg;
		cvtColor(blendMask, currentImg, COLOR_GRAY2BGR);
		maskGaussianPyramid.push_back(currentImg); //highest level

		currentImg = blendMask;
		for (int l = 1; l<levels + 1; l++) {
			Mat _down;
			if (leftLapPyr.size() > l) {
				pyrDown(currentImg, _down, leftLapPyr[l].size());
			}
			else {
				pyrDown(currentImg, _down, leftSmallestLevel.size()); //smallest level
			}

			Mat down;
			cvtColor(_down, down, COLOR_GRAY2BGR);
			maskGaussianPyramid.push_back(down);
			currentImg = _down;
		}
	}

	void buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel) {
		lapPyr.clear();
		Mat currentImg = img;
		for (int l = 0; l<levels; l++) {
			Mat down, up;
			pyrDown(currentImg, down);
			pyrUp(down, up, currentImg.size());
			Mat lap = currentImg - up;
			lapPyr.push_back(lap);
			currentImg = down;
		}
		currentImg.copyTo(smallestLevel);
	}

	Mat_<Vec3f> reconstructImgFromLapPyramid() {
		Mat currentImg = resultSmallestLevel;
		for (int l = levels - 1; l >= 0; l--) {
			Mat up;

			pyrUp(currentImg, up, resultLapPyr[l].size());
			currentImg = up + resultLapPyr[l];
		}
		return currentImg;
	}

	void blendLapPyrs() {
		resultSmallestLevel = leftSmallestLevel.mul(maskGaussianPyramid.back()) +
			rightSmallestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());
		for (int l = 0; l<levels; l++) {
			Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
			Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
			Mat B = rightLapPyr[l].mul(antiMask);
			Mat_<Vec3f> blendedLevel = A + B;

			resultLapPyr.push_back(blendedLevel);
		}
	}

public:
	LaplacianBlending(const Mat_<Vec3f>& _left, const Mat_<Vec3f>& _right, const Mat_<float>& _blendMask, int _levels) :
		left(_left), right(_right), blendMask(_blendMask), levels(_levels)
	{
		assert(_left.size() == _right.size());
		assert(_left.size() == _blendMask.size());
		buildPyramids();
		blendLapPyrs();
	};

	Mat_<Vec3f> blend() {
		return reconstructImgFromLapPyramid();
	}
};

Mat_<Vec3f> LaplacianBlend(const Mat_<Vec3f>& l, const Mat_<Vec3f>& r, const Mat_<float>& m) {
	LaplacianBlending lb(l, r, m, 4);
	return lb.blend();
}


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
    Mat input_1 = imread(meta_parser["data"][0], IMREAD_COLOR); //Load as grayscale
    Mat input_2 = imread(meta_parser["data"][1], IMREAD_COLOR); //Load as grayscale

    resize(input_1, input_1, Size(input_1.size().width / 6, input_1.size().height / 6), cv::INTER_AREA);
    resize(input_2, input_2, Size(input_2.size().width / 6, input_2.size().height / 6), cv::INTER_AREA);

    Mat output; // output of sift
    get_keypoints(input_1, kpts_image_1, desc_1);
    get_keypoints(input_2, kpts_image_2, desc_2);

    // Add results to image and save.
    //show_keypoints(input_1, output, kpts_image_1);
    //imshow("matches", output);

    std::vector<Point2f> kpts_1;
    std::vector<Point2f> kpts_2;
    std::vector<float> keypoints_distance;

    // We would also create a matric of size nxn where n is number images.
    // Then the (i,j)th element should be the normalized distanace value.
    // This could generate a symmetrical matrix.
    // And then in each row, we see which  element is lowest, the lowest element index will be matched image.
    // say the lowest element is a{p,q} then pth image matches qth image best and must be stitched...
    // Now I am stuck on what should be the order ....

    // Show matched keypoints
    vector<DMatch> matches;
    match(desc_1, desc_2, matches);

    Mat img_matches;
    drawMatches(input_1, kpts_image_1, input_2, kpts_image_2, matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("matched_image", img_matches);

    vector<char> match_mask(matches.size(), 1);
    keypoints_distance.reserve(matches.size());
    if (static_cast<int>(match_mask.size()) < 3)
    {
        cout << "Not enough correspondence";
        return 0;
    }
    for (int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        kpts_1.push_back(kpts_image_1[matches[i].queryIdx].pt);
        kpts_2.push_back(kpts_image_2[matches[i].trainIdx].pt);
        keypoints_distance.push_back(matches[i].distance);
    }

    //Matched images will have least normalized distance.
    float normalized_distance = std::accumulate(keypoints_distance.begin(), keypoints_distance.end(), 0) / keypoints_distance.size();

    // We must find a way to keep cache this distance, then attach images based on this distance.
    std::cout << "normalized_distance: " << normalized_distance << std::endl;

    Mat H = findHomography(kpts_1, kpts_2, RANSAC);
    cout << "Keypoint 1 size = " << kpts_1.size() << " Keypoint_2_size = " << kpts_2.size() << endl;
    cout << "Homography matrix  : " << H << endl;

    cv::Mat result;
    warpPerspective(input_2, result, H.inv(), cv::Size(input_1.cols + input_2.cols, input_2.rows), INTER_CUBIC);
    // imshow("Result_warped", result);

	//BlendLaplacian(input_1, result);

    cv::Mat half(result, cv::Rect(0, 0, input_1.cols, input_1.rows));
    input_1.copyTo(half);
    imshow("Result Panorama", result);


    waitKey(0);
    return 0;
}


int BlendLaplacian(Mat l8u, Mat r8u) {
	Mat_<Vec3f> l; l8u.convertTo(l, CV_32F, 1.0 / 255.0);
	Mat_<Vec3f> r; r8u.convertTo(r, CV_32F, 1.0 / 255.0);

	Mat_<float> m(l.rows, l.cols, 0.0);
	m(Range::all(), Range(0, m.cols / 2)) = 1.0;

	Mat_<Vec3f> blend = LaplacianBlend(l, r, m);
	imshow("blended", blend);
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