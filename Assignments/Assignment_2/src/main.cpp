#include "main.h"

int main(int argc, const char *argv[])
{
#ifdef _MSC_VER
    std::ifstream ifile("C:/Projects/Acads/COL780/Assignments/Assignment_2/input/meta.json");
#else
    std::ifstream ifile("Assignments/Assignment_2/input/meta.json");
#endif
    vector<vector<KeyPoint>> keypoint_all_img;
    vector<vector<Point2f>> keypoint_as_point2f_all_img;
    vector<Mat> descriptors_all_img;
    json meta_parser;
    ifile >> meta_parser;
    vector<Mat> all_images;
    int index = (int)meta_parser["testcase"]["run_case"];
    populate_images_from_dir(meta_parser["testcase"]["filename"][index], all_images);
    sample_down(all_images);
    get_keypoints_and_descriptors_for_all_imgs(all_images, keypoint_all_img, descriptors_all_img);
    // show_keypoints(all_images[0], all_images[0], keypoint_all_img[0]);
    map<pair<int, int>, float> distances;

    for (size_t i = 0; i < all_images.size(); i++)
    {
        for (size_t j = i + 1; j < all_images.size(); j++)
        {
            vector<DMatch> matches;
            match(all_images[i], all_images[j], matches);
            float accumulate = 0.0f;
            for (size_t k = 0; k < static_cast<int>(matches.size()); k++)
            {
                accumulate += matches[i].distance;
            }
            accumulate /= matches.size();
            distances[make_pair(i, j)] = accumulate;
        }
    }

    // print distances
    for (auto i = distances.begin(); i != distances.end(); i++)
    {
        cout << "<" << std::get<0>(i->first) + 1 << ", " << std::get<1>(i->first) + 1 << ">"
             << " = " << i->second << endl;
    }

    // distances[make_pair(1, 2)] = 3.4f;
    // auto it = distances.find(make_pair(1, 3));
    // if (it != distances.end())
    // {
    //     std::cout << distances[make_pair(1, 2)];
    // }

    imshow("img", all_images[0]);
    waitKey(0);
}
