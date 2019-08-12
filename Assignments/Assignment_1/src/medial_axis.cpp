#include "medial_axis.h"

using namespace cv;


void MedialAxis_C::GetMedialAxis(cv::Mat& frame_bw, std::vector<Vec2f>& lines, Point2d& medial_axis, Point& object_center, double& largest_eigen_value)
{
	// Get Detected line points
	std::vector<Point> line_points;
	for (size_t i = 0; i < lines.size(); i++) {
		// std::cout << " No of Lines " << lines.size() << "\n";
		float rho = lines[i][0];
		float theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 500 * (-b));
		pt1.y = cvRound(y0 + 500 * (a));
		pt2.x = cvRound(x0 - 500 * (-b));
		pt2.y = cvRound(y0 - 500 * (a));

		line(frame_bw, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);

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
		object_center = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)), static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

		medial_axis = Point2d(pca_analysis.eigenvectors.at<double>(0, 0), pca_analysis.eigenvectors.at<double>(0, 1));

		// The first eigen value is the largest
		largest_eigen_value = pca_analysis.eigenvalues.at<double>(0);
	}
}

// Seems like an unnecessary function
void MedialAxis_C::TransformColors(cv::Mat& in_frame, Mat& out_frame, cv::ColorConversionCodes color_code)
{
	cvtColor(in_frame, out_frame, color_code);
}

void MedialAxis_C::DetectLines(Mat& frame, Mat& out_frame, std::vector<Vec2f>& detected_lines)
{
	cv::Mat resized_frame;
	resize(frame, frame, Size(_image_width, _image_height)); //todo: Is it not important to pass on different mats?

#ifdef DUMP_IMAGES
	imwrite("C:/Projects/Acads/out/1.jpg", frame);
#endif

	// Convert to grey scale
	cv::Mat grey_frame;
	TransformColors(frame, grey_frame, COLOR_BGR2GRAY);

#ifdef DUMP_IMAGES
	imwrite("C:/Projects/Acads/out/2.jpg", grey_frame);
#endif
	// Histogram equalization
	if (_apply_histogram_equalization) {
		equalizeHist(grey_frame, grey_frame);
	}

	// Get foreground image
	cv::Mat fg_image;
	_pBackSub->apply(grey_frame, fg_image);


#ifdef DUMP_IMAGES
	imwrite("C:/Projects/Acads/out/fg_image.jpg", fg_image);
#endif

	// Opening morph operation on foreground image
	cv::Mat filtered_image;
	Mat element = getStructuringElement(_morph_elem, Size(2 * _morph_size + 1, 2 * _morph_size + 1), Point(_morph_size, _morph_size)); // TUNE
	morphologyEx(fg_image, filtered_image, _operation, element);


#ifdef DUMP_IMAGES
	imwrite("C:/Projects/Acads/out/morphed_filtered_image.jpg", fg_image);
#endif

	//Thresholding to get rid of shadows
	if (_apply_thresholding) {
		threshold(filtered_image, filtered_image,_threshold, _threshold_max, THRESH_BINARY);
	}


	// A round of gauss blur
	blur(filtered_image, filtered_image, Size(_blur_size, _blur_size));// Mostly isotropic blurring is used


#ifdef DUMP_IMAGES
	imwrite("C:/Projects/Acads/out/before_canny_filtered_image.jpg", filtered_image);
#endif
	// Canny edge detector
	Canny(filtered_image, filtered_image, _canny_low, _canny_high, 3);


#ifdef DUMP_IMAGES
	imwrite("C:/Projects/Acads/out/fina_filtered_image.jpg", filtered_image);
#endif

	// Hough Transform to fit line
	HoughLines(filtered_image, detected_lines, 1, CV_PI / 180, _hough_thresold, 0, 0);
	
	// copy the processed frame for rendering.
	filtered_image.copyTo(out_frame);
}
