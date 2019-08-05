// Current project
#include "EigenExample.h"

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// Eigen
#include <Eigen/Dense>

// STL includes
#include <iostream>
#include <string.h>

using Eigen::MatrixXd;
using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
	Mat src = imread(argv[1], IMREAD_GRAYSCALE);
	if (src.empty()) {
		cout << "I Could not open or find the image!\n"
			<< endl;
		cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}

	namedWindow("Display window", WINDOW_AUTOSIZE);

	imshow("Display window", src);
	
    waitKey(0);
    return 0;
}
