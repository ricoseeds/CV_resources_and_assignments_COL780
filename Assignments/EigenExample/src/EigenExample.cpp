#include "EigenExample.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// Eigen includes
#include <Eigen/Dense>

// STL includes
#include <iostream>

using Eigen::MatrixXd;
using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    char the_path[256];

    getcwd(the_path, 255);
    strcat(the_path, "/");
    strcat(the_path, argv[0]);

    printf("%s\n", the_path);
    cv::Mat original = imread("data/lena.jpg", 0);
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", original);
    waitKey(0);
    return 0;
}
