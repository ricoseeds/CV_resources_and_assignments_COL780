#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "json.hpp"

// cv headers
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;
using json = nlohmann::json;

typedef enum
{
    gray_to_rgb,
    rgb_to_gray
} ColorSpace;

typedef enum
{
    opening = 2
} MorphType;

static int IMG_HEIGHT, IMG_WIDTH;

//functions
void TransformColors(Mat &, Mat &, ColorSpace);

void CreateSplitWindow(Mat &, Mat &, Mat &);

void DrawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2);


int main(int argc, char **argv)
{
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();

#ifdef _MSC_VER
    std::ifstream ifile("C:/Projects/Acads/COL780/Assignments/Assignment_1/input/tuning_params.json");
#else
	std::ifstream ifile("Assignments/Assignment_1/input/tuning_params.json");
#endif

    json j;
    ifile >> j;
    // declarations
    const string filename = j["data"];
    // VideoCapture capture(filename);
    VideoCapture capture(filename);
    if (!capture.isOpened())
    {
        cerr << "Unable to open video file" << endl;
        return 0;
    }
    IMG_HEIGHT = j["resize_height"];
    IMG_WIDTH = j["resize_width"];
    Mat frame, frame_bw, mog2_mask;
    Mat concatenated_window_frame(cv::Size(IMG_WIDTH * 2, IMG_HEIGHT), CV_8UC3);
    int morph_elem = j["morph_element"];    // 0
    int morph_size = j["morph_point_size"]; // 3
    int operation = MORPH_OPEN;
    //Sobel
    Mat grad_x, grad_y, grad;
    Mat abs_grad_x, abs_grad_y;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    //Hough
    vector<Vec2f> lines;

    while (true)
    {
        capture >> frame;
        if (frame.empty())
            break;
        namedWindow("Display window", 0);
        // resize frame optional
        resize(frame, frame, Size(IMG_WIDTH, IMG_HEIGHT));
        // convert to gray scale
        TransformColors(frame, frame_bw, rgb_to_gray);
        // Histogram Equilization <NOT REQUIRED>
        if (j["bin_thresh"])
        {
            equalizeHist(frame_bw, frame_bw);
        }
        // background sub
        pBackSub->apply(frame_bw, frame_bw);

        // opening morph operation on fgMask
        Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size)); // TUNE
        morphologyEx(frame_bw, frame_bw, operation, element);

        //Thresholding to get rid of shadows
        if (j["bin_thresh"])
        {
            threshold(frame_bw, frame_bw, j["img_min_thresh"], j["img_max_thresh"], 0);
        }

        // A round of gauss blur
        blur(frame_bw, frame_bw, Size(j["gaussian_kernel_size"], j["gaussian_kernel_size"]));

        // Canny edge detector
        Canny(frame_bw, frame_bw, j["canny_low_threshold"], j["canny_high_threshold"], 3);

        // Hough Transform to fit line
        HoughLines(frame_bw, lines, 1, CV_PI / 180, j["hough_threshold"], 0, 0); // runs the actual detection


		// Get Detected line points
		std::vector<Point> line_points;
        for (size_t i = 0; i < lines.size(); i++)
        {
            // std::cout << " No of Lines " << lines.size() << "\n";
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 500 * (-b));
            pt1.y = cvRound(y0 + 500 * (a));
            pt2.x = cvRound(x0 - 500 * (-b));
            pt2.y = cvRound(y0 - 500 * (a));

#ifdef DISPLAY_LINES
            line(frame, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
#endif
			// Get all line points
			// Another approach can be get all points from the image that are near to these detected lines.
			LineIterator it(frame_bw, pt1, pt2, 8);
			for (int i = 0; i < it.count; i++, ++it) {
				line_points.push_back(it.pos());
			}

        }

		// Compute PCA
		vector<Point2d> eigen_vecs(2);
		vector<double> eigen_val(2);

		if (!line_points.empty()) {
			// std::cout << line_points.size() << std::endl;
			//Perform PCA analysis
			Mat data_pts = Mat(line_points.size(), 2, CV_64F);
			for (int i = 0; i < data_pts.rows; i++) {
				data_pts.at<double>(i, 0) = line_points[i].x;
				data_pts.at<double>(i, 1) = line_points[i].y;
			}

			PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
			//Store the center of the object
			Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)), static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

			// Store the eigenvalues and eigenvectors
			for (int i = 0; i < 2; i++)
			{
				eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0), pca_analysis.eigenvectors.at<double>(i, 1));
				// std::cout << eigen_vecs[i] << std::endl;
				eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
				// std::cout << eigen_val[i] << std::endl;
			}


			Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
			DrawAxis(frame, cntr, p1, Scalar(0, 255, 0), 1);
		}

		// Clear lines and points
		line_points.clear();
        lines.clear();

        // Rendering stage
        TransformColors(frame_bw, frame_bw, gray_to_rgb); // mandatory step so that concatenated_window_frame has same color Channel
        CreateSplitWindow(frame, frame_bw, concatenated_window_frame);

        // show frame
        imshow("Display window", concatenated_window_frame);
        int keyboard = waitKey(1); // ?
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
}

void TransformColors(Mat &frame, Mat &output_fr, ColorSpace c) // TODO : add all conversions for transforming color space
{
    switch (c)
    {
    case gray_to_rgb:
        cvtColor(frame, output_fr, cv::COLOR_GRAY2RGB);
        break;
    case rgb_to_gray:
        cvtColor(frame, output_fr, cv::COLOR_BGR2GRAY);
        break;
    }
}

void CreateSplitWindow(Mat &m1, Mat &m2, Mat &ccat_frame)
{
    m1.copyTo(ccat_frame(Rect(0, 0, IMG_WIDTH, IMG_HEIGHT)));
    m2.copyTo(ccat_frame(Rect(IMG_WIDTH, 0, IMG_WIDTH, IMG_HEIGHT)));
}

// A temperory method, we might not need it.
void DrawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale)
{
	double angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
	double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	// Here we lengthen the arrow by a factor of scale
	q.x = (int)(p.x - scale * hypotenuse * cos(angle));
	q.y = (int)(p.y - scale * hypotenuse * sin(angle));
	line(img, p, q, colour, 1, LINE_AA);
	// create the arrow hooks
	p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
	line(img, p, q, colour, 1, LINE_AA);
	p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
	line(img, p, q, colour, 1, LINE_AA);
}
