#include "medial_axis.h"

using namespace cv;


void MedialAxis_C::GetMedialAxis(cv::Mat& frame_bw, std::vector<Vec2f>& lines, Point2d& medial_axis, Point2d& object_center)
{
	// Get Detected line points
	std::vector<Point> line_points;
	for (size_t i = 0; i < lines.size(); i++)
	{
		// std::cout << " No of Lines " << lines.size() << "\n";
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 500 * (-b));
		pt1.y = cvRound(y0 + 500 * (a));
		pt2.x = cvRound(x0 - 500 * (-b));
		pt2.y = cvRound(y0 - 500 * (a));

		// Get all line points
		// Another approach can be get all 
		// points from the image that are near to these detected lines.
		LineIterator it(frame_bw, pt1, pt2, 8);
		for (int i = 0; i < it.count; i++, ++it) {
			line_points.push_back(it.pos());
		}

	}

	if (!line_points.empty()) {
		// std::cout << line_points.size() << std::endl;
		//Perform PCA analysis
		Mat data_pts = Mat(line_points.size(), 2, CV_64F);
		for (int i = 0; i < data_pts.rows; i++) {
			data_pts.at<double>(i, 0) = line_points[i].x;
			data_pts.at<double>(i, 1) = line_points[i].y;
		}

		PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
		
		//Store the center of the object
		object_center = Point2d(pca_analysis.mean.at<double>(0, 0),pca_analysis.mean.at<double>(0, 1));

		medial_axis = Point2d(pca_analysis.eigenvectors.at<double>(0, 0), pca_analysis.eigenvectors.at<double>(0, 1));
	}
}

// Seems like an unnecessary function
void MedialAxis_C::TransformColors(cv::Mat& in_frame, cv::Mat& out_frame, cv::ColorConversionCodes color_code)
{
	cvtColor(in_frame, out_frame, color_code);
}

void MedialAxis_C::DetectLines(Mat& frame, std::vector<Vec2f>& detected_lines)
{
	cv::Mat resized_frame;
	resize(frame, resized_frame, Size(_image_width, _image_height)); //todo: Is it not important to pass on the 


	// Convert to grey scale
	cv::Mat grey_frame;
	TransformColors(resized_frame, grey_frame, COLOR_BGR2GRAY);


	// Histogram equalization
	if (_apply_histogram_equalization) {
		equalizeHist(grey_frame, grey_frame);
	}


	// Create MOG2 background subractor
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorMOG2();


	// Get foreground image
	cv::Mat fg_image;
	pBackSub->apply(grey_frame, fg_image);


	// Opening morph operation on foreground image
	cv::Mat filtered_image;
	Mat element = getStructuringElement(_morph_elem, Size(2 * _morph_size + 1, 2 * _morph_size + 1), Point(_morph_size, _morph_size)); // TUNE
	morphologyEx(fg_image, filtered_image, _operation, element);


	//Thresholding to get rid of shadows
	if (_apply_thresholding) {
		threshold(filtered_image, filtered_image,_threshold, _threshold_max, THRESH_BINARY);
	}


	// A round of gauss blur
	blur(filtered_image, filtered_image, Size(_blur_size, _blur_size));// Mostly isotropic blurring is used


	// Canny edge detector
	Mat edge_mask;
	Canny(filtered_image, edge_mask, _canny_low, _canny_high, 3);


	// Hough Transform to fit line
	HoughLines(edge_mask, detected_lines, 1, CV_PI / 180, _hough_thresold, 0, 0);

}
