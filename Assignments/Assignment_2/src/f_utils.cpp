#include "f_utils.h"

void populate_images_from_dir(std::string relative_path, std::vector<cv::Mat> &all_images)
{
    boost::filesystem::path p(relative_path);
    if (is_directory(p))
    {
        for (auto &entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {}))
        {
            cv::Mat input = cv::imread(entry.path().string(), 0);
            all_images.push_back(input);
        }
    }
}
