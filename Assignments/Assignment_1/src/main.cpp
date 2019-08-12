// Project includes
#include "medial_axis.h"

// Jason parser
#include "json.hpp"

using namespace cv;
using namespace std;
using json = nlohmann::json;


namespace
{

// Helper functions
void CreateSplitWindow(Mat& frame, Mat& out_frame, Mat& matDst)
{
	cvtColor(out_frame, out_frame, COLOR_GRAY2BGR);
	Mat matRoi = matDst(Rect(0, 0, frame.cols, frame.rows));
	frame.copyTo(matRoi);
	matRoi = matDst(Rect(frame.cols, 0, frame.cols, frame.rows));
	out_frame.copyTo(matRoi);
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

}


int main(int argc, char **argv)
{
#ifdef _MSC_VER
    std::ifstream ifile("C:/Projects/Acads/COL780/Assignments/Assignment_1/input/tuning_params.json");
#else
	std::ifstream ifile("Assignments/Assignment_1/input/tuning_params.json");
#endif

	// Get algorithm configuration
    json algo_configs;
    ifile >> algo_configs;
    
	// Declarations
    const string filename = algo_configs["data"];
    
	// Create MOG2 background subractor
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorMOG2();


    VideoCapture capture(filename);
    if (!capture.isOpened())
    {
        cerr << "Unable to open video file" << endl;
        return 0;
    }
    
	// We may use Sobel as well.
	// Mat grad_x, grad_y, grad;
	// Mat abs_grad_x, abs_grad_y;
	// int scale = 1;
	// int delta = 0;
	// int ddepth = CV_16S;

	Mat frame;
    vector<Vec2f> lines;

	// Create the object of Medial Axis detection algorithm
	MedialAxis_C medial_axis(algo_configs["resize_width"], algo_configs["resize_height"],
		algo_configs["morph_element"], algo_configs["morph_point_size"], MORPH_OPEN,
		algo_configs["img_min_thresh"], algo_configs["img_max_thresh"],
		algo_configs["canny_low_threshold"], algo_configs["canny_high_threshold"],
		algo_configs["hough_threshold"]);

	medial_axis.SetBackGroundSubtractor(pBackSub);

	// Algorithmic pipeline settings for optional algorithms.
	medial_axis.SetHistogramEqualization(false);
	medial_axis.SetThresholding(false);

    while (true) {
        capture >> frame;
		if ( frame.empty() ) { 
			break;
		}

        namedWindow("Display window", 0);

		// Main processing block
		{
			// Detect lines and also get the processed frame as output.
			Mat out_frame;
			medial_axis.DetectLines(frame, out_frame, lines);

			// Find Medial axis of line
			if (!lines.empty()) {
				Point2d medial_vector(0, 0);
				Point object_center(0, 0);
				double largest_eigen_value = 0.0;

				medial_axis.GetMedialAxis(frame, lines, medial_vector, object_center, largest_eigen_value);

				Point p1 = object_center + 0.02 * Point(static_cast<int>(medial_vector.x * largest_eigen_value), static_cast<int>(medial_vector.y * largest_eigen_value));
				DrawAxis(frame, object_center, p1, Scalar(0, 255, 0), 1);

				// Clear lines
				lines.clear();
			}

			// Rendering stage
			Mat matDst(Size(frame.cols * 2, frame.rows), frame.type(), Scalar::all(0));
			CreateSplitWindow(frame, out_frame, matDst);

			// show frame
			imshow("Display window", matDst);
		
		}
		
        int keyboard = waitKey(1); // ?
		if ( keyboard == 'q' || keyboard == 27 ) {
			break;
		}
        
    }
}