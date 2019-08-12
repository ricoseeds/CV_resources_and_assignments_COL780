#pragma once


#include "json.hpp"

// cv headers
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>


#include <iostream>
#include <sstream>
#include <fstream>
#include <string>


class MedialAxis_C {

public : // Constructor/ Destructor 

	// Delete constructors not required.
	MedialAxis_C() = default;
	MedialAxis_C(const MedialAxis_C&) = delete;
	MedialAxis_C& operator=(const MedialAxis_C&) = delete;

	MedialAxis_C(unsigned int image_width, unsigned int image_height,
		unsigned int morph_elem, unsigned int morph_size,
		unsigned int morph_operation, double threshold,
		double threshold_max, double canny_low, double canny_high, int hough_thresold) :
		_image_width(image_width), _image_height(image_height), _morph_elem(morph_elem),
		_morph_size(morph_size), _operation(morph_operation),
		_threshold(threshold), _threshold_max(threshold_max),
		_canny_low(canny_low), _canny_high(canny_high), _hough_thresold(hough_thresold){}


private : // Internal methods
	
	void TransformColors(cv::Mat& in_frame, cv::Mat& out_frame, cv::ColorConversionCodes color_code);

public: // Public Methods


	void SetHistogramEqualization(bool flag) { _apply_histogram_equalization = flag; }

	void SetThresholding(bool flag) { _apply_thresholding = flag; }

	void SetBackGroundSubtractor(cv::Ptr<cv::BackgroundSubtractor> pBackSub) { _pBackSub = pBackSub; }

	void DetectLines(cv::Mat& frame, cv::Mat& out_frame, std::vector<cv::Vec2f>& detected_lines);

	void GetMedialAxis(cv::Mat& frame_bw, std::vector<cv::Vec2f>& lines, cv::Point2d& medial_axis, cv::Point& object_center, double& largest_eigen_value);

private : // class attributes

	// image related
	unsigned int _image_height = 1000;
	unsigned int _image_width = 1000;

	// morphology related
	unsigned int _morph_elem = 0;
	unsigned int _morph_size = 0;
	int _operation = cv::MORPH_OPEN;

	bool _apply_histogram_equalization = false;
	bool _apply_thresholding = false;

	// Image thresholding params
	double _threshold = 0.0;
	double _threshold_max = 255.0;

	// Gaussian blur var
	unsigned int _blur_size{5};

	// Canny edge related
	double _canny_low = 100.0;
	double _canny_high = 200.0;

	int _hough_thresold = 50;

	cv::Ptr<cv::BackgroundSubtractor> _pBackSub = nullptr;

};