#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <iostream>
#include <string>

// prototypes
void populate_images_from_dir(std::string relative_path, std::vector<cv::Mat> &all_images);
