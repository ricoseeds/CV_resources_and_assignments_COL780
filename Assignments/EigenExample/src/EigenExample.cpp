#include "EigenExample.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char **argv)
{
    char the_path[256];

    getcwd(the_path, 255);
    strcat(the_path, "/");
    strcat(the_path, argv[0]);

    printf("%s\n", the_path);
    cv::Mat original = imread("data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", original);
    waitKey(0);
    return 0;
}
