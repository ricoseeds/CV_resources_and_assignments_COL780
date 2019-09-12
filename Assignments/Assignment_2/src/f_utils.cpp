#include "f_utils.h"

void populate_images_from_dir(std::string relative_path, std::vector<cv::Mat> &all_images)
{
    std::vector<std::string> filenames;
    cv::glob(relative_path, filenames);
    for (size_t i = 0; i < filenames.size(); ++i)
    {
        cv::Mat input = cv::imread(filenames[i], 0); // Set 0 for B/W images
        all_images.push_back(input);
    }
}
