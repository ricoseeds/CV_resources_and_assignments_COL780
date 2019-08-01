#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
    //! [Load image]
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty())
    {
        cout << "I Could not open or find the image!\n"
             << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    //! [Load image]

    //! [Convert to grayscale]
    cvtColor(src, src, COLOR_BGR2GRAY);
    //! [Convert to grayscale]

    //! [Apply Histogram Equalization]
    Mat dst;
    equalizeHist(src, dst);
    //! [Apply Histogram Equalization]

    //! [Display results]
    imshow("Source image", src);
    imshow("Equalized Image", dst);
    //! [Display results]

    //! [Wait until user exits the program]
    waitKey();
    //! [Wait until user exits the program]

    return 0;
}
