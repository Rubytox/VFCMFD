/**
 * @file    surf.cpp
 * This file implements the fucntions defined
 * in surf.hpp
 */

#include "../include/surf.hpp"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

/**
 * This function only detects SURF keypoints and 
 * draws them on the source image.
 *
 * @param   src         The source image.
 * @param   minHessian  The value of the Hessian threshold in the SURF algorithm.
 *
 * @return  A matrix representing the source image with the SURF keypoints
 *          drawn on it.
 */
[[deprecated]] Mat surf_analysis(const Mat& src, int minHessian)
{
    Ptr<SURF> detector = SURF::create(minHessian);
    vector<KeyPoint> keypoints;
    detector->detect(src, keypoints);

    Mat dst;
    drawKeypoints(src, keypoints, dst);

    return dst;
}

/**
 * Helper function that computes a similarity vector for a keypoint.
 * Given \f$X = \{x_1, ..., x_n\}\f$ a set of keypoints and \f$F = \{f_1, ..., f_n\}\f$
 * their respective descriptors, and given an index \f$i \in [1, n]\f$, this function
 * computes the following vector:
 * \f$D = \{d_1, ..., d_{n-1}\}\f$ such as :
 * - \f$\forall j \in [1, n]\setminus\lbrace i\rbrace,\; d_j = ||f_i - f_j||_2\f$
 * - the coordinates of \f$D\f$ are sorted by ascending order
 *
 * The code actually returns a map because as we need to sort the distances, we need to
 * keep the correspondance between a computed distance and the matching descriptor. 
 * Thus, we return a map associating distances with the matching keypoint's index.
 *
 * @param   keypoints       The list of keypoints.
 * @param   descriptors     A matrix whose dimensions are len(keypoints) x 64 or len(keypoints) x 128.
 *                          The i-th line of the matrix is the descriptor of the i-th keypoint in keypoints.
 * @param   i               The index of the keypoint we want to compute a similarity vector for.
 *
 * @return  A map associating the euclidean distance between the i-th keypoint and all the other
 *          keypoints and the index of the other keypoint.
 */
[[deprecated]] map<double, int> similarityVector(const vector<KeyPoint>& keypoints, const Mat& descriptors, int i)
{
    /*
     * This map will contain pairs <euclidean_distance, keypoint_index>
     *
     * I put euclidean distance as the key because a map is ordered by keys
     */
    map<double, int> distances;

    Mat descriptor = descriptors.row(i);
    
    for (int j = 0; j < keypoints.size(); j++) {
        if (i != j) {
            Mat other = descriptors.row(j);

            double distance = norm(descriptor, other, NORM_L2);

            distances.insert({ distance, j });
        }
    }
    
    return distances;
}

/*
 * Returns random number in [a, b[
 *
 * @condition a < b
 */
[[deprecated]] int randRange(int a, int b)
{
    return rand() % (b - a) + a;
}

[[deprecated]] Mat transformAndDraw(const Mat& img, vector<Line> lines, bool print = false)
{
    /* Mat dst = fullPlane(img, lines); */
    Mat dst = img.clone();

    int length = img.cols;
    int height = img.rows;
    for (auto const& droite : lines) {
        droite.draw(dst);
        droite.prolonger(dst);
        Line&& ortho = droite.getOrtho();
        /* ortho.translate(length, height); */
        ortho.draw(dst, Scalar(0xFF, 0, 0));
        if (print)
            cout << droite << endl;
    }

    return dst;
}

[[deprecated]] Mat parallelLines(const Mat& img)
{
    Mat dst(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

    vector<Line> lines;

    int minX = 0;
    int maxX = img.cols;
    
    int minY = 0;
    int maxY = img.rows;

    for (int j = 0; j < 4; j++) {
        Point2f point1(randRange(minX, maxX), randRange(minY, maxY));
        Point2f point2(randRange(minX, maxX), randRange(minY, maxY));
        Line initial(point1, point2);
        lines.push_back(initial);

        int step = 100;
        for (int i = 1; i < 10; i++) {
            Line newLine(initial);
            newLine.translate(step * i, step * i);
            lines.push_back(newLine);
        }
    }


    return transformAndDraw(dst, lines, true);
}

/**
 * This function implements the algorithm presented in the paper:
 * "A SIFT-Based Forensic Method for Copy–Move Attack Detection 
 *  and Transformation Recovery".
 * It detects SURF keypoints using the regular SURF algorithm, but the
 * matching process is customized in order to fit our needs : it is
 * done on only one image instead of two.
 *
 * @param  img         The source image.
 * @param  minHessian  The value of the Hessian threshold in the SURF algorithm.
 *
 * @param  A matrix representing the source image, with matches between similar
 *          areas drawn.
 */
[[deprecated]] Mat surf_improved(const Mat& img, int minHessian)
{
    double threshold = 0.5;

    Ptr<SURF> detector = SURF::create(minHessian);
    vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    /* cout << "Starting matching process" << endl; */
    /* cout << "KeyPoints: " << keypoints.size() << endl; */

    vector<vector<int>> allMatches;
    /* cout << "Looking for matches for keypoint no "; */
    for (int i = 0; i < keypoints.size(); i++) {
        /* cout << i; */
        /* cout.flush(); */
        map<double, int> similarity = similarityVector(keypoints, descriptors, i);

        map<double, int>::iterator it_1;
        map<double, int>::iterator it_2;
        vector<int> matches;
        for (it_1 = similarity.begin(),
             it_2 = it_1,
             it_2++;
             it_1 != similarity.end() && it_2 != similarity.end();
             it_1++, it_2++) {
            double distance_1 = it_1->first;
            double distance_2 = it_2->first;
            if (distance_1 / distance_2 < threshold)
                matches.push_back(it_1->second);
            else
                break;
        }
        allMatches.push_back(matches);

        /* string nb = to_string(i); */
        /* for (auto const& c : nb) */
        /*     cout << '\b'; */
    }
    /* cout << endl; */

    int longueur_min = 30;
    vector<Line> droites;
    for (int i = 0; i < allMatches.size(); i++) {
        vector<int> matches = allMatches[i];
        Point2f origin = keypoints[i].pt;
        for (const auto& index : matches) {
            Point2f other = keypoints[index].pt;
            Line droite(origin, other);
            if (droite.length() >= longueur_min)
                droites.push_back(droite);
        }
    }

    for (const auto& droite : droites) {
        cout << droite << endl;
    }

    Mat lines = img.clone();
    for (const auto& droite : droites) {
        droite.draw(lines, Scalar(0, 0, 0xFF), 1);
    }

    return lines;

    /* return transformAndDraw(img, droites, true); */
}

/**
 * This function detects SURF keypoints on two images, and then
 * computes the matching keypoints between both images.
 *
 * Note: This function should be used when img2 is part of img1.
 *
 * @param   img1        The source image.
 * @param   img2        An image usually extracted from img1. It can be applied
 *                      transformations such as rotations, scaling...
 * @param   minHessian  The value of the Hessian threshold in the SURF algorithm.
 *
 * @return  A matrix representing the two images side by side, with matches
 *          between the keypoints drawn.
 */
[[deprecated]] Mat surf_matching(const Mat& img1, const Mat& img2,
                  int minHessian)
{
    Ptr<SURF> detector = SURF::create(minHessian);
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptor1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptor2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    /* vector<vector<DMatch>> matches; */
    /* matcher->radiusMatch(descriptor1, descriptor2, matches, 0.1); */
    vector<DMatch> matches;
    matcher->match(descriptor1, descriptor1, matches);

    Mat img_matches = img1.clone();
    /* int max = matches.size() >= 30 ? 30 : matches.size(); */
    /* vector<DMatch> reduced_matches = vector<DMatch>(matches.begin(), matches.begin() + max); */
    /* drawMatches(img1, keypoints1, img2, keypoints2, reduced_matches, img_matches); */
    /* drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches); */

    vector<DMatch> filtered_matches;
    for (int i = 0; i < matches.size(); i++) {
        DMatch match = matches[i];
        Point2f point1 = keypoints1[match.queryIdx].pt;
        Point2f point2 = keypoints1[match.trainIdx].pt;
        if (areNotNear(point1, point2)) {
        /* if (point1.x != point2.x || point1.y != point2.y) { */
            filtered_matches.push_back(match);
        }
    }

    /* drawMatches(img1, keypoints1, img2, keypoints2, filtered_matches, img_matches); */

    for (auto & match : filtered_matches) {
        Point2f point1 = keypoints1[match.queryIdx].pt;
        Point2f point2 = keypoints1[match.trainIdx].pt;
        // Format of Scalar color is BGR
        line(img_matches, point1, point2, Scalar(0x0, 0xFF, 0x0), 3);
    }

    return img_matches;
}

/**
 * Helper function used in surf_matching algorithm: it checks
 * whether two points are near to each other.
 * In this function, p1 is near to p2 if p1 is one of the eight neighbours
 * of p2 or is equal to p2 :
 *
 * \verbatim
 +-----+-----+-----+
 |  a  |  b  |  c  |
 +-----+-----+-----+
 |  d  | p_2 |  e  |  p1 is near to p2 <==> p1 == p2 or p1 in [a, h]
 +-----+-----+-----+
 |  f  |  g  |  h  |
 +-----+-----+-----+
 \endverbatim
 *
 * Note: this relation is symmetric.
 *
 * @param   point1  The first point.
 * @param   point2  The second point.
 *
 * @return  True if point1 is not near to point2, false otherwise.
 */
[[deprecated]] bool areNotNear(const Point2f& point1, const Point2f& point2)
{
    return point1.x < point2.x - 1
        || point1.x > point2.x + 1
        || point1.y < point2.y - 1
        || point1.y > point2.y + 1;
}
