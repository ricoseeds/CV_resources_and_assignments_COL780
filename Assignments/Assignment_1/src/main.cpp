#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char **argv)
{

    cv::Mat original = cv::imread("data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", original);
    cv::waitKey(0);
    return 0;
}
