#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "json.hpp"

// cv headers
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
void convert_to_gray_scale(Mat &, Mat &, ColorSpace);
void show_split_window(Mat &, Mat &, Mat &);
int main(int argc, char **argv)
{
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();
    std::ifstream ifile("Assignments/Assignment_1/input/tuning_params.json");
    json j;
    ifile >> j;
    // declarations
    const string filename = j["data"];
    // VideoCapture capture(filename);
    VideoCapture capture("data/videos/1.mp4");
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
    int operation = MorphType::opening;
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
        convert_to_gray_scale(frame, frame_bw, rgb_to_gray);
        // Histogram Equilization <NOT REQUIRED>
        // equalizeHist(frame_bw, frame_bw);
        // background sub
        pBackSub->apply(frame_bw, frame_bw);

        // opening morph operation on fgMask
        Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size)); // TUNE
        morphologyEx(frame_bw, frame_bw, operation, element);

        //Thresholding to get rid of shadows
        threshold(frame_bw, frame_bw, j["img_min_thresh"], j["img_max_thresh"], 0);

        // A round of gauss blur
        blur(frame_bw, frame_bw, Size(j["gaussian_kernel_size"], j["gaussian_kernel_size"]));

        // Canny edge detector
        Canny(frame_bw, frame_bw, j["canny_low_threshold"], j["canny_high_threshold"], 3);

        // Hough Transform to fit line
        // vector<Vec2f> lines;                                     // will hold the results of the detection
        HoughLines(frame_bw, lines, 1, CV_PI / 180, 20, 30, 1); // runs the actual detection

        for (size_t i = 0; i < lines.size(); i++)
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            line(frame_bw, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
        }
        lines.clear();
        // Rendering stage
        convert_to_gray_scale(frame_bw, frame_bw, gray_to_rgb); // mandatory step so that concatenated_window_frame has same color Channel
        show_split_window(frame, frame_bw, concatenated_window_frame);

        // show frame
        imshow("Display window", concatenated_window_frame);
        int keyboard = waitKey(1); // ?
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
}

void convert_to_gray_scale(Mat &frame, Mat &output_fr, ColorSpace c) // TODO : add all conversions for transforming color space
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

void show_split_window(Mat &m1, Mat &m2, Mat &ccat_frame)
{
    m1.copyTo(ccat_frame(Rect(0, 0, IMG_WIDTH, IMG_HEIGHT)));
    m2.copyTo(ccat_frame(Rect(IMG_WIDTH, 0, IMG_WIDTH, IMG_HEIGHT)));
    // imshow("Display window", ccat_frame);
}