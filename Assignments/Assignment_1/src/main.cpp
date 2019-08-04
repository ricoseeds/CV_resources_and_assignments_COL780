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
    int lowThreshold = 100; // 100
    const int ratio = 3;
    const int kernel_size = 3;
    Mat detected_edges, dst;
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

        // Canny edge detection on (fgMask => GRAY_CHANNEL)
        dst.create(fgMask.size(), fgMask.type()); // MIGHTBEBUG:  we might need color channed
        blur(fgMask, detected_edges, Size(3, 3));
        Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
        dst = Scalar::all(0);
        fgMask.copyTo(dst, detected_edges);

        // change color channel of mask
        cvtColor(dst, fgMask_col, cv::COLOR_GRAY2RGB);

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
