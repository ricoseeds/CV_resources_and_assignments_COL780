#include "main.h"

int main(int argc, const char *argv[])
{
#ifdef _MSC_VER
    std::ifstream ifile("C:/Projects/Acads/COL780/Assignments/Assignment_2/input/meta.json");
#else
    std::ifstream ifile("Assignments/Assignment_2/input/meta.json");
#endif
    vector<vector<KeyPoint>> keypoint_all_img;
    vector<Mat> descriptors_all_img;
    json meta_parser;
    ifile >> meta_parser;
    vector<Mat> all_images;
    int index = (int)meta_parser["testcase"]["run_case"];
    populate_images_from_dir(meta_parser["testcase"]["filename"][index], all_images);
    sample_down(all_images);
    get_keypoints_and_descriptors_for_all_imgs(all_images, keypoint_all_img, descriptors_all_img);
    // show_keypoints(all_images[0], all_images[0], keypoint_all_img[0]);
    imshow("img", all_images[0]);
    waitKey(0);
}
