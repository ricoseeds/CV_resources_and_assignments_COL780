#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
// const char *params = "{ help h         |                   | Print usage }"
//                      "{ input          | ../data/vtest.avi | Path to a video or a sequence of image }"
//                      "{ algo           | MOG2              | Background subtraction method (KNN, MOG2) }";
int main(int argc, char **argv)
{
    const int IMG_HEIGHT = 480, IMG_WIDTH = 640;
    int morph_elem = 0;     // 0
    int morph_size = 2;     // 3
    int morph_operator = 0; // 0
    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();
    // pBackSub = createBackgroundSubtractorKNN();
    VideoCapture capture("data/videos/1.mp4");
    if (!capture.isOpened())
    {
        //error in opening the video input
        cerr << "Unable to open video file" << endl;
        return 0;
    }
    Mat frame, fgMask, fgMask_col;
    Mat win_mat(cv::Size(IMG_WIDTH * 2, IMG_HEIGHT), CV_8UC3);
    while (true)
    {
        capture >> frame;
        if (frame.empty())
            break;
        //update the background model
        pBackSub->apply(frame, fgMask);

        // // Copy small images into big mat
        cv::resize(frame, frame, cv::Size(IMG_WIDTH, IMG_HEIGHT));
        cv::resize(fgMask, fgMask, cv::Size(IMG_WIDTH, IMG_HEIGHT));

        //get the frame number and write it on the current frame
        rectangle(frame, cv::Point(10, 2), cv::Point(100, 20), cv::Scalar(255, 255, 255), -1);
        stringstream ss;
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        //show the current frame and the fg masks
        namedWindow("Display window", 0); // Create a window for display.

        // opening morph operation on fgMask
        int operation = morph_operator + 2;
        Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
        morphologyEx(fgMask, fgMask, operation, element);

        // Canny edge detection on fgMask

        // change color channel for display purpose
        cvtColor(fgMask, fgMask_col, cv::COLOR_GRAY2RGB);

        // show operation : window
        frame.copyTo(win_mat(cv::Rect(0, 0, IMG_WIDTH, IMG_HEIGHT)));
        fgMask_col.copyTo(win_mat(cv::Rect(IMG_WIDTH, 0, IMG_WIDTH, IMG_HEIGHT)));
        imshow("Display window", win_mat);

        //get the input from the keyboard
        int keyboard = waitKey(1); // ?
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
}

// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
// #include <iostream>
// using namespace cv;
// Mat src, src_gray;
// Mat dst, detected_edges;
// int lowThreshold = 0;
// const int max_lowThreshold = 100;
// const int ratio = 3;
// const int kernel_size = 4;
// const char *window_name = "Edge Map";
// static void CannyThreshold(int, void *)
// {
//     blur(src_gray, detected_edges, Size(3, 3));
//     Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
//     dst = Scalar::all(0);
//     src.copyTo(dst, detected_edges);
//     imshow(window_name, dst);
// }
// int main(int argc, char **argv)
// {
//     // CommandLineParser parser(argc, argv, "{@input | ../data/fruits.jpg | input image}");
//     src = imread("data/filtered_probe.png", IMREAD_COLOR); // Load an image
//     if (src.empty())
//     {
//         std::cout << "Could not open or find the image!\n"
//                   << std::endl;
//         std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
//         return -1;
//     }
//     dst.create(src.size(), src.type());
//     cvtColor(src, src_gray, COLOR_BGR2GRAY);
//     namedWindow(window_name, WINDOW_AUTOSIZE);
//     createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
//     CannyThreshold(0, 0);
//     waitKey(0);
//     return 0;
// }