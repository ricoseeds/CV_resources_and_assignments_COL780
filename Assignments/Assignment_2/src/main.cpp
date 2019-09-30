#include "main.h"
void get_Dij_by_distances_of_matched_inliers(vector<Mat> &all_images, map<pair<int, int>, vector<DMatch>> &image_i_j_matches, map<pair<int, int>, Mat> &image_i_j_homography_mask, map<pair<int, int>, float> &distances);
void get_Dij_by_match_count(vector<Mat> &all_images, map<pair<int, int>, vector<DMatch>> &image_i_j_matches, map<pair<int, int>, Mat> &image_i_j_homography_mask, map<pair<int, int>, pair<int, int>> &match_count);
void get_match_fraction(map<pair<int, int>, pair<int, int>> match_count, map<pair<int, int>, float> &match_fraction);
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
    vector<Mat> all_images_color;
    int index = (int)meta_parser["testcase"]["run_case"];
    populate_images_from_dir(meta_parser["testcase"]["filename"][index], all_images);
    populate_images_from_dir_color(meta_parser["testcase"]["filename"][index], all_images_color);
    kMaxMatchingSize = meta_parser["kMaxMatchingSize"];
    sample_down(all_images);
    sample_down(all_images_color);
    get_keypoints_and_descriptors_for_all_imgs(all_images, keypoint_all_img, descriptors_all_img);
    // show_keypoints(all_images[0], all_images[0], keypoint_all_img[0]);
    map<pair<int, int>, float> distances;
    map<pair<int, int>, float> match_fraction;
    map<pair<int, int>, pair<int, int>> match_count;
    map<pair<int, int>, vector<DMatch>>
        image_i_j_matches;
    map<pair<int, int>, vector<Point2f>> image_i_j_matches_point2f_query;
    map<pair<int, int>, vector<Point2f>> image_i_j_matches_point2f_train;
    map<pair<int, int>, Mat> image_i_j_homography;
    map<pair<int, int>, Mat> image_i_j_homography_mask;
    map<pair<int, int>, Mat> image_i_j_homography_result;
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
        Mat H = findHomography(kpts_a, kpts_b, RANSAC, meta_parser["ransac_re_proj_threshold"], hmask, 4000, 0.998);
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
        get_match_fraction(match_count, match_fraction);
        // Declaring the type of Predicate that accepts 2 pairs and return a bool
        typedef std::function<bool(std::pair<pair<int, int>, float>, std::pair<pair<int, int>, float>)> Comparator;

        // Defining a lambda function to compare two pairs. It will compare two pairs using second field
        Comparator compFunctor =
            [](std::pair<pair<int, int>, float> elem1, std::pair<pair<int, int>, float> elem2) {
                return elem1.second >= elem2.second;
            };
        std::set<std::pair<pair<int, int>, float>, Comparator> match_fract_set(
            match_fraction.begin(), match_fraction.end(), compFunctor);
        cout << " \n\n ---- \n\n";
        match_fraction.clear();
        int count = 0;
        // generate graph
        vector<vector<int>> G;
        G.resize(all_images.size(), std::vector<int>(all_images.size(), 0));
        for (auto &i : match_fract_set)
        {
            // cout << "<" << std::get<0>(i.first) + 1 << ", " << std::get<1>(i.first) + 1 << ">"
            //      << " = " << i.second << endl;
            match_fraction[make_pair(get<0>(i.first), get<1>(i.first))] = i.second;
            if (count++ < all_images.size() - 1)
            {
                G[(int)std::get<0>(i.first)][(int)std::get<1>(i.first)] = 1;
                G[(int)std::get<1>(i.first)][(int)std::get<0>(i.first)] = 1;
                cout << "<" << std::get<0>(i.first) << ", " << std::get<1>(i.first) << ">"
                     << " = " << i.second << endl;
            }
            // G[(int)std::get<0>(i.first)][(int)std::get<1>(i.first)] = 1.0 - i.second;
            // G[(int)std::get<1>(i.first)][(int)std::get<0>(i.first)] = 1.0 - i.second;
            // cout << G[std::get<0>(i.first)][std::get<1>(i.first)];
        }
        cout << "graph : " << endl;
        for (size_t i = 0; i < G.size(); i++)
        {
            for (size_t j = 0; j < G[i].size(); j++)
            {
                cout << G[i][j] << " ";
            }
            cout << endl;
        }
        int source = max_deg_row(G);
        cout << "SOURCE = " << source << endl;
        vector<int> visited(G.size(), 0);
        vector<int> rejection_list;
        map<int, vector<int>> result_map;
        is_connected_from_source(G, visited, source, rejection_list);
        cout << "Visited list " << endl;
        for (size_t i = 0; i < visited.size(); i++)
        {
            cout << visited[i] << " ";
        }
        cout << endl;
        cout << "/nRejection list " << endl;
        // add source to rejection list
        rejection_list.push_back(source);
        for (size_t i = 0; i < rejection_list.size(); i++)
        {
            cout << rejection_list[i] << " ";
        }
        cout << endl;
        bfs(G, source, result_map);
        for (size_t i = 0; i < result_map.size(); i++)
        {
            cout << " <" << i << "> = { ";
            for (size_t j = 0; j < result_map[i].size(); j++)
            {
                cout << result_map[i][j] << " ";
            }
            cout << " } " << endl;
        }
        // Do stitching with a given source

        for (size_t i = 0; i < all_images.size(); i++)
        {
            if (!(std::find(rejection_list.begin(), rejection_list.end(), i) != rejection_list.end()))
            {
                // InDirect homography
                Mat H = Mat::eye(3, 3, CV_64F);
                if (result_map[i].size() > 2)
                {
                    Mat H_next = Mat::eye(3, 3, CV_64F);
                    for (size_t k = result_map[i].size() - 1; k > 0; k--)
                    {
                        map<pair<int, int>, Mat>::iterator iter = image_i_j_homography.find(make_pair(result_map[i][k], result_map[i][k - 1]));
                        if (iter != image_i_j_homography.end())
                        {
                            H_next = image_i_j_homography[make_pair(result_map[i][k], result_map[i][k - 1])];
                        }
                        else
                        {
                            H_next = image_i_j_homography[make_pair(result_map[i][k - 1], result_map[i][k])].inv();
                        }
                        H = H_next * H;
                    }
                    image_i_j_homography_result[make_pair(source, result_map[i][0])] = H.inv();
                }
                else
                {
                    map<pair<int, int>, Mat>::iterator iter = image_i_j_homography.find(make_pair(source, i));
                    if (iter != image_i_j_homography.end())
                    {
                        H = image_i_j_homography[make_pair(source, i)];
                    }
                    else
                    {
                        H = image_i_j_homography[make_pair(i, source)].inv();
                    }
                    image_i_j_homography_result[make_pair(source, i)] = H.inv();
                }
            }
        }
        // cout << "HEREEEE";

        // Do stitching
        cv::Mat black_img(cv::Size(2000, 1000), CV_32FC3, Scalar(0));
        Mat result_referece_img = black_img;
        Mat T = Mat::eye(3, 3, CV_64FC1);
        T.at<double>(0, 2) = result_referece_img.cols / 2;
        T.at<double>(1, 2) = result_referece_img.rows / 2;
        warpPerspective(all_images_color[source], result_referece_img, T, Size(result_referece_img.cols, result_referece_img.rows), INTER_LINEAR, BORDER_CONSTANT, 0);
        // imshow("Reference", result_referece_img);
        int c = 0;
        Mat blended_padded;
        vector<Mat> homified_images;
        homified_images.push_back(result_referece_img);
        for (auto i = image_i_j_homography_result.begin(); i != image_i_j_homography_result.end(); i++)
        {
            int img_2 = get<1>(i->first);
            Mat H = cv::Mat(cv::Size(3, 3), CV_64FC1);
            H = image_i_j_homography_result[make_pair(source, img_2)];
            Mat tmp;
            warpPerspective(all_images_color[img_2], tmp, T * H, Size(black_img.cols, black_img.rows), INTER_LINEAR, BORDER_CONSTANT, 0);
            homified_images.push_back(tmp);
        }
        Mat master_image(cv::Size(2000, 1000), CV_8UC3);
        cout << "MASTERTYPE " << master_image.type();
        cout << "HOMOIFIED " << homified_images[0].type();
        for (size_t i = 0; i < black_img.rows; i++)
        {
            for (size_t j = 0; j < black_img.cols; j++)
            {
                Vec3b acc = Vec3b(0, 0, 0);
                for (size_t k = 0; k < homified_images.size(); k++)
                {
                    Vec3b pix = homified_images[k].at<Vec3b>(i, j);
                    acc += pix;
                }
                master_image.at<Vec3b>(i, j) = acc;
            }
        }
        imshow("RESULT", master_image);
        // write to file -- to be deleted
        for (size_t i = 0; i < homified_images.size(); i++)
        {
            imwrite("/Users/arghachakraborty/Projects/CV_assignments/data/homified_3/test" + to_string(i) + ".jpg", homified_images[i]);
            // imshow("dasda", homified_images[i]);
        }
    }
    else
    {
        cerr << "Choose [1 | 2] in the 'matching_heuristics' field of meta.json";
    }

    // // print distances
    // for (auto i = distances.begin(); i != distances.end(); i++)
    // {
    //     cout << "<" << std::get<0>(i->first) + 1 << ", " << std::get<1>(i->first) + 1 << ">"
    //          << " = " << i->second << endl;
    // }

    // imshow("img", all_images[0]);
    waitKey(0);
}

void get_match_fraction(map<pair<int, int>, pair<int, int>> match_count, map<pair<int, int>, float> &match_fraction)
{
    for (auto i = match_count.begin(); i != match_count.end(); i++)
    {
        match_fraction[make_pair(std::get<0>(i->first), std::get<1>(i->first))] = (float)(std::get<0>(i->second) / (float)std::get<1>(i->second));
        cout << "<" << std::get<0>(i->first) + 1 << ", " << std::get<1>(i->first) + 1 << ">"
             << " = " << match_fraction[make_pair(std::get<0>(i->first), std::get<1>(i->first))] << endl;
    }
}
void get_Dij_by_match_count(vector<Mat> &all_images, map<pair<int, int>, vector<DMatch>> &image_i_j_matches, map<pair<int, int>, Mat> &image_i_j_homography_mask, map<pair<int, int>, pair<int, int>> &match_count)
{
    for (size_t i = 0; i < all_images.size(); i++)
    {
        for (size_t j = i + 1; j < all_images.size(); j++)
        {
            match_count[make_pair(i, j)] = make_pair(countNonZero(image_i_j_homography_mask[make_pair(i, j)]), image_i_j_homography_mask[make_pair(i, j)].size().height);
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