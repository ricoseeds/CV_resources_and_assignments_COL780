#include "main.h"
void get_Dij_by_distances_of_matched_inliers(vector<Mat> &all_images, map<pair<int, int>, vector<DMatch>> &image_i_j_matches, map<pair<int, int>, Mat> &image_i_j_homography_mask, map<pair<int, int>, float> &distances);
void get_Dij_by_match_count(vector<Mat> &all_images, map<pair<int, int>, vector<DMatch>> &image_i_j_matches, map<pair<int, int>, Mat> &image_i_j_homography_mask, map<pair<int, int>, pair<int, int>> &match_count);
int main(int argc, const char *argv[])
{
    // RANSAC uses random number hence we need a seed to get consistent results
    theRNG().state = 1234;
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
    map<pair<int, int>, pair<int, int>> match_count;
    map<pair<int, int>, vector<DMatch>>
        image_i_j_matches;
    map<pair<int, int>, vector<Point2f>> image_i_j_matches_point2f_query;
    map<pair<int, int>, vector<Point2f>> image_i_j_matches_point2f_train;
    map<pair<int, int>, Mat> image_i_j_homography;
    map<pair<int, int>, Mat> image_i_j_homography_mask;

    for (size_t i = 0; i < all_images.size(); i++)
    {
        for (size_t j = i + 1; j < all_images.size(); j++)
        {
            vector<DMatch> matches;
            match(descriptors_all_img[i], descriptors_all_img[j], matches);
            image_i_j_matches[make_pair(i, j)] = matches;
        }
    }
    // compute keypoint in point2f
    for (auto i = image_i_j_matches.begin(); i != image_i_j_matches.end(); i++)
    {
        vector<Point2f> kpts_a, kpts_b;
        int index_i, index_j;
        index_i = std::get<0>(i->first);
        index_j = std::get<1>(i->first);
        for (size_t j = 0; j < i->second.size(); j++)
        {
            kpts_a.push_back(keypoint_all_img[index_i][i->second[j].queryIdx].pt);
            kpts_b.push_back(keypoint_all_img[index_j][i->second[j].trainIdx].pt);
        }
        image_i_j_matches_point2f_query[make_pair(index_i, index_j)] = kpts_a;
        image_i_j_matches_point2f_train[make_pair(index_i, index_j)] = kpts_b;
        Mat hmask;
        // Good result for high values of ransac_re_proj_threshold
        Mat H = findHomography(kpts_a, kpts_b, RANSAC, meta_parser["ransac_re_proj_threshold"], hmask, 2000, 0.998);
        image_i_j_homography[make_pair(index_i, index_j)] = H;
        image_i_j_homography_mask[make_pair(index_i, index_j)] = hmask;
    }

    // Build Dij metric
    if (meta_parser["matching_heuristics"] == 1)
    {
        get_Dij_by_distances_of_matched_inliers(all_images, image_i_j_matches, image_i_j_homography_mask, distances);
    }
    else if (meta_parser["matching_heuristics"] == 2)
    {
        get_Dij_by_match_count(all_images, image_i_j_matches, image_i_j_homography_mask, match_count);
    }
    else
    {
        cerr << "Choose [1 | 2] in the 'matching_heuristics' field of meta.json";
    }

    // print distances
    for (auto i = distances.begin(); i != distances.end(); i++)
    {
        cout << "<" << std::get<0>(i->first) + 1 << ", " << std::get<1>(i->first) + 1 << ">"
             << " = " << i->second << endl;
    }

    imshow("img", all_images[0]);
    waitKey(0);
}

void get_Dij_by_match_count(vector<Mat> &all_images, map<pair<int, int>, vector<DMatch>> &image_i_j_matches, map<pair<int, int>, Mat> &image_i_j_homography_mask, map<pair<int, int>, pair<int, int>> &match_count)
{
    for (size_t i = 0; i < all_images.size(); i++)
    {
        for (size_t j = i + 1; j < all_images.size(); j++)
        {
            // cout << " [" << i + 1 << " , " << j + 1 << " ] = " << countNonZero(image_i_j_homography_mask[make_pair(i, j)]) << " Size = " << image_i_j_homography_mask[make_pair(i, j)].size().height << endl;
            // cout << " [" << i << " , " << j << " ] = " << image_i_j_homography_mask[make_pair(i, j)] << endl;
        }
    }
}

// Really big signature, needs refactor
void get_Dij_by_distances_of_matched_inliers(vector<Mat> &all_images, map<pair<int, int>, vector<DMatch>> &image_i_j_matches, map<pair<int, int>, Mat> &image_i_j_homography_mask, map<pair<int, int>, float> &distances)
{
    for (size_t i = 0; i < all_images.size(); i++)
    {
        for (size_t j = i + 1; j < all_images.size(); j++)
        {
            vector<DMatch> matches;
            matches = image_i_j_matches[make_pair(i, j)];
            float accumulate = 0.0f;
            for (size_t k = 0; k < static_cast<int>(matches.size()); k++)
            {
                // kill the outliers
                if (image_i_j_homography_mask[make_pair(i, j)].at<int>(0, k))
                {
                    accumulate += matches[i].distance;
                }
            }
            accumulate /= matches.size();
            distances[make_pair(i, j)] = accumulate;
        }
    }
}