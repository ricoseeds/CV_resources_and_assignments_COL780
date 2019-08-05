#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <iostream>
#include <sstream>
using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    const int IMG_HEIGHT = 480, IMG_WIDTH = 640;
    int morph_elem = 0;     // 0
    int morph_size = 1;     // 3
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
        // TODO: Histogram Equilization

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
        namedWindow("Display window", WINDOW_NORMAL); // Create a window for display.

        // opening morph operation on fgMask
        int operation = morph_operator + 2;
        Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size)); // TUNE
        morphologyEx(fgMask, fgMask, operation, element);

        //SHOW RESULT OF MORPH
        cvtColor(fgMask, fgMask_col, cv::COLOR_GRAY2RGB);

        // TODO: Use sobel with tuning on its kernel size

        // TODO: Hough transform

        // SAVE: Canny edge detection on (fgMask => GRAY_CHANNEL)
        // dst.create(fgMask.size(), fgMask.type());
        // blur(fgMask, detected_edges, Size(3, 3)); // TUNE
        // Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
        // dst = Scalar::all(0);
        // fgMask.copyTo(dst, detected_edges);

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
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/highgui.hpp"
// #include <iostream>
// using namespace cv;
// using namespace std;
// int main(int argc, char **argv)
// {
//     cv::CommandLineParser parser(argc, argv,
//                                  "{@input   |data/filtered_probe.png|input image}"
//                                  "{ksize   k|1|ksize (hit 'K' to increase its value)}"
//                                  "{scale   s|1|scale (hit 'S' to increase its value)}"
//                                  "{delta   d|0|delta (hit 'D' to increase its value)}"
//                                  "{help    h|false|show help message}");
//     cout << "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n";
//     parser.printMessage();
//     cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";
//     // First we declare the variables we are going to use
//     Mat image, src, src_gray;
//     Mat grad;
//     const String window_name = "Sobel Demo - Simple Edge Detector";
//     int ksize = parser.get<int>("ksize");
//     int scale = parser.get<int>("scale");
//     int delta = parser.get<int>("delta");
//     int ddepth = CV_16S;
//     String imageName = parser.get<String>("@input");
//     // As usual we load our source image (src)
//     image = imread(imageName, IMREAD_COLOR); // Load an image
//     // Check if image is loaded fine
//     if (image.empty())
//     {
//         printf("Error opening image: %s\n", imageName.c_str());
//         return 1;
//     }
//     for (;;)
//     {
//         // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
//         GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
//         // Convert the image to grayscale
//         cvtColor(src, src_gray, COLOR_BGR2GRAY);
//         Mat grad_x, grad_y;
//         Mat abs_grad_x, abs_grad_y;
//         Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
//         Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
//         // converting back to CV_8U
//         convertScaleAbs(grad_x, abs_grad_x);
//         convertScaleAbs(grad_y, abs_grad_y);
//         addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
//         Canny(grad, grad, 50, 200, 3);
//         imshow(window_name, grad);
//         char key = (char)waitKey(0);
//         if (key == 27)
//         {
//             return 0;
//         }
//         if (key == 'k' || key == 'K')
//         {
//             ksize = ksize < 30 ? ksize + 2 : -1;
//         }
//         if (key == 's' || key == 'S')
//         {
//             scale++;
//         }
//         if (key == 'd' || key == 'D')
//         {
//             delta++;
//         }
//         if (key == 'r' || key == 'R')
//         {
//             scale = 1;
//             ksize = -1;
//             delta = 0;
//         }
//     }
//     return 0;
// }

// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/imgproc.hpp"
// using namespace cv;
// using namespace std;
// int main(int argc, char **argv)
// {
//     // Declare the output variables
//     Mat dst, cdst, cdstP;
//     const char *default_file = "data/sobel_canny.png";
//     const char *filename = argc >= 2 ? argv[1] : default_file;
//     // Loads an image
//     Mat src = imread(filename, IMREAD_GRAYSCALE);
//     // Check if image is loaded fine
//     if (src.empty())
//     {
//         printf(" Error opening image\n");
//         printf(" Program Arguments: [image_name -- default %s] \n", default_file);
//         return -1;
//     }
//     // Edge detection
//     Canny(src, dst, 10, 50, 3);
//     // Copy edges to the images that will display the results in BGR
//     cvtColor(dst, cdst, COLOR_GRAY2BGR);
//     cdstP = cdst.clone();
//     // Standard Hough Line Transform
//     vector<Vec2f> lines;                               // will hold the results of the detection
//     HoughLines(dst, lines, 1, CV_PI / 180, 255, 0, 0); // runs the actual detection
//     // Draw the lines
//     for (size_t i = 0; i < lines.size(); i++)
//     {
//         float rho = lines[i][0], theta = lines[i][1];
//         Point pt1, pt2;
//         double a = cos(theta), b = sin(theta);
//         double x0 = a * rho, y0 = b * rho;
//         pt1.x = cvRound(x0 + 1000 * (-b));
//         pt1.y = cvRound(y0 + 1000 * (a));
//         pt2.x = cvRound(x0 - 1000 * (-b));
//         pt2.y = cvRound(y0 - 1000 * (a));
//         // line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
//     }
//     // Probabilistic Line Transform
//     // vector<Vec4i> linesP;                                 // will hold the results of the detection
//     // HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
//     // // Draw the lines
//     // for (size_t i = 0; i < linesP.size(); i++)
//     // {
//     //     Vec4i l = linesP[i];
//     //     line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
//     // }
//     // Show results
//     imshow("Source", src);
//     imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
//     // imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
//     // Wait and Exit
//     waitKey();
//     return 0;
// }
