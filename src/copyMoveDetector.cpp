//
// Created by rubytox on 10/06/2020.
//

#include "../include/copyMoveDetector.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace defals;


/**
 * Creates a copy/move forgery detector object. We pass the hessian and matches threshold
 * here.
 *
 * TODO: - minHessian might be computed depending on image size or characteristics.
 *         For instance, we could compute the keypoints (this is fast usually), then
 *         with some statistics check whether we have enough keypoints.
 *       - matchesThreshold should be a constant as specified in the research paper.
 *         We need to check though whether 0.5 is a good enough value.
 *
 * @param filename          The name of the image file.
 * @param masqueName        The name of the falsification binary mask.
 *                          Can be empty.
 * @param minHessian        The threshold value of Det(Hessian).
 * @param matchesThreshold  The maximum ratio for two consecutives distances in matching algorithm.
 */
copyMoveDetector::copyMoveDetector(const DetectorOptions& options) : _options(options) {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _copyMoveDetector_ constructor";

    BOOST_LOG_TRIVIAL(debug) << "Reading file " << options.image;
    _image = cv::imread(options.image, cv::IMREAD_COLOR);
    if (_image.empty()) {
        cerr << "Couldn't find file " << options.image << endl;
        exit(1);
    }

    BOOST_LOG_TRIVIAL(debug) << "Mask provided: " << boolalpha << !options.mask.empty() << noboolalpha;
    if (!options.mask.empty()) {
        BOOST_LOG_TRIVIAL(debug) << "Reading file " << options.mask;
        _mask = cv::imread(options.mask, cv::IMREAD_GRAYSCALE);
        if (_mask.empty()) {
            cerr << "Couldn't find file " << options.mask << endl;
            exit(1);
        }
    }


    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _copyMoveDetector_ constructor";
}

/**
 * This function runs all the detecting process by :
 * - computing keypoints in the image
 * - computing all matches between keypoints using
 *   improved g2NN algorithm
 * - creating all the lines from the matches and having
 *   them all in the same direction
 * - sorting the lines in clusters
 * - computing convex hulls out of the clusters
 * - computing mask out of the convex hulls
 * - extending the mask using EQM expansion
 * - computing the Dice index if a binary mask is provided
 */
void copyMoveDetector::detect() {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _detect_";

    computeKeypoints();
    computeBetterMatches();
    computeLines();
    computeClusters();
    computeHull();

    computeMask();
    extendMask();

    double dice = computeDice();
    if (dice != -1)
        BOOST_LOG_TRIVIAL(info) << "Dice: " << dice;

    double precision = 0, recall = 0, F1 = 0;
    computeFScore(precision, recall, F1);
    if (precision != -1 && recall != -1 && F1 != -1) {
        BOOST_LOG_TRIVIAL(info) << "Precision: " << precision;
        BOOST_LOG_TRIVIAL(info) << "Recall: " << recall;
        BOOST_LOG_TRIVIAL(info) << "F1-Score: " << F1;
    }

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _detect_";
}

/**
 * This function prints all the lines in
 * rho,theta,length format.
 * This may be useful for further processing.
 */
[[deprecated]] void copyMoveDetector::printLines() const {
    for (const auto& line : _lines) {
        cerr << line << endl;
    }
}

/**
 * This function prints all the clusters in
 * noCluster,rho,theta,length format.
 * This may be useful for further processing.
 */
[[deprecated]] void copyMoveDetector::printClusters() const {
    int n = _clusters.size();
    cout << n << endl;
    for (int i = 0; i < n; i++) {
        for (const auto& line : _clusters[i])
            cout << i << ',' << line << endl;
    }
}

/**
 * This function prints information about the detection.
 * It should be used for debug purposes only.
 *
 * It prints the matched lines in the format:
 *              x,y,theta,norm
 * where (x,y) is the start point, theta the angle difference between
 * the matched points and norm is the descriptor norm difference between
 * the matched points.
 */
[[debug_only]] void copyMoveDetector::printInfo() {
    vector<double> diffAngle, diffNorm;

    for (const auto& cluster : _clusters) {
        for (const auto &line : cluster) {
            const InterestPoint &start = line.getPoint1();
            const InterestPoint &end = line.getPoint2();

            int idx1 = start.getAngleIdx();
            int idx2 = end.getAngleIdx();

            double angleDiff = start.angle - end.angle;
            double normDiff = norm(start.getDescriptor()) - norm(end.getDescriptor());
            cout << "cluster," << idx1 << "," << idx2 << "," << angleDiff << "," << normDiff << endl;
        }
    }

    for (const auto& line : _lines) {
        const InterestPoint &start = line.getPoint1();
        const InterestPoint &end = line.getPoint2();

        int idx1 = start.getAngleIdx();
        int idx2 = end.getAngleIdx();

        double angleDiff = start.angle - end.angle;
        double normDiff = norm(start.getDescriptor()) - norm(end.getDescriptor());
        cout << idx1 << "," << idx2 << "," << angleDiff << "," << normDiff << endl;
    }
}

/**
 * Shows all the steps taken by the algorithm to detect
 * copy-move forgeries.
 *
 * @param keypoints     If set to true, keypoints are drawn on the image and saved.
 * @param lines         If set to true, lines are drawn on the image and saved.
 * @param clusters      If set to true, clusters are drawn on the image and saved.
 * @param hulls         If set to true, convex hulls are drawn on the image and saved.
 * @param maskName      The name of the binary mask output file.
 *                      Can be empty.
 * @param mask          If set to true, the aforementionned steps are drawn as well on
 *                      the binary mask.
 *
 */
void copyMoveDetector::show() const {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _show_";

    Mat dst_mask = _mask.clone();

    Mat comparison = Mat::zeros(_image.size(), CV_8UC3);
    if (!_options.mask.empty() && !_hulls.empty()) {
        Vec3b original(0, 0xFF, 0);
        Vec3b extended(0, 0, 0xFF);
        for (int x = 0; x < _image.cols; x++) {
            for (int y = 0; y < _image.rows; y++) {
                uchar intensity_original = _mask.at<uchar>(y, x);
                uchar intensity_extended = _extendedMask.at<uchar>(y, x);
                Vec3b teinte;
                if (intensity_original == 0xFF)
                    teinte += original;
                if (intensity_extended == 0xFF)
                    teinte += extended;
                comparison.at<Vec3b>(y, x) = teinte;
            }
        }
    }

    if (_options.draw_kp) {
        BOOST_LOG_TRIVIAL(debug) << "Drawing keypoints";
        Mat kp_canvas = _image.clone();
        drawKeypoints(_image, _interestPoints.asKeyPoints(), kp_canvas);
        save(kp_canvas, _options.rawName + "_0keypoints.jpg");
    }

    if (_options.draw_matches) {
        BOOST_LOG_TRIVIAL(debug) << "Drawing matches";
        Mat lines_canvas = _image.clone();
        for (const auto &line : _lines) {
            line.draw(lines_canvas, Scalar(0), 1);
            if (!_options.mask.empty())
                line.draw(dst_mask, Scalar(0xFF, 0xFF, 0xFF), 1);
        }
        save(lines_canvas, _options.rawName + "_1matches.jpg");
    }

    if (_options.draw_clusters) {
        BOOST_LOG_TRIVIAL(debug) << "Drawing clusters";
        Mat cluster_canvas = _image.clone();

        int i = 0;
        for (const auto& cluster : _clusters) {
            unsigned int R = rand() / ((RAND_MAX + 1u) / 255);
            unsigned int G = rand() / ((RAND_MAX + 1u) / 255);
            unsigned int B = rand() / ((RAND_MAX + 1u) / 255);
            Scalar color(B, G, R);

            BOOST_LOG_TRIVIAL(trace) << "Cluster n°" << i << "'s color: " << R << " " << G << " " << B;

            for (const auto& line : cluster) {
                line.draw(cluster_canvas, color, 1);
                if (!_options.mask.empty())
                    line.draw(dst_mask, color, 1);
            }

            i++;
        }

        save(cluster_canvas, _options.rawName + "_2clusters.jpg");
    }

    if (_options.draw_hulls) {
        BOOST_LOG_TRIVIAL(debug) << "Drawing convex hulls";
        Mat hulls_canvas = _image.clone();

        vector<vector<Point>> hullsList;
        for (const auto& hull : _hulls) {
            vector<Point> pts;
            for (const auto& pt : hull) {
                pts.push_back(pt.first.pt);
            }
            hullsList.push_back(pts);
        }
        for (size_t i = 0; i < _hulls.size(); i++) {
            unsigned int R = rand() / ((RAND_MAX + 1u) / 255);
            unsigned int G = rand() / ((RAND_MAX + 1u) / 255);
            unsigned int B = rand() / ((RAND_MAX + 1u) / 255);
            Scalar color(B, G, R);

            BOOST_LOG_TRIVIAL(trace) << "Convex hull n°" << i << "'s color: " << R << " " << G << " " << B;

            drawContours(hulls_canvas, hullsList, i, color, FILLED);
            if (!_options.mask.empty())
                drawContours(dst_mask, hullsList, i, color, FILLED);
        }
        save(hulls_canvas, _options.rawName + "_3hulls.jpg");
    }


    if (!_options.mask.empty()) {
        BOOST_LOG_TRIVIAL(debug) << "Saving masks";
        save(dst_mask, _options.rawName + "_4mask.png");
        save(comparison, _options.rawName + "_5mask_comparison.jpg");
    }

    if (!_extendedMask.empty())
        save(_extendedMask, _options.rawName + "_binary_mask.jpg");
    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _show_";
}

/**
 * This function is called by _show(const string&, const bool, const bool, const bool)_.
 *
 * @param img       The canvas to display.
 * @param filename  The filename to save the canvas.
 */
inline void copyMoveDetector::save(const Mat &img, const std::string& filename) {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _save_";

    BOOST_LOG_TRIVIAL(debug) << "Saving image under name: " << filename;
    if (!imwrite(filename, img))
        cout << "Couldn't save target image" << endl;

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _save_";
}

/**
 * This function applies the SURF algorithm in order to
 * find keypoints for the image.
 */
void copyMoveDetector::computeKeypoints() {
    BOOST_LOG_TRIVIAL(info) << "Entering _computeKeypoints_";

    BOOST_LOG_TRIVIAL(debug) << "Creating SURF detector with minHessian = " << _options.kp_hessian;
    Ptr<SURF> detector = SURF::create(_options.kp_hessian);
    vector<KeyPoint> keypoints;
    detector->detect(_image, keypoints);

    Mat descriptors;
    detector->compute(_image, keypoints, descriptors);
    _interestPoints = InterestPoints(keypoints, descriptors, _options.g2NN_angleThreshold, _options.g2NN_normThreshold);

    BOOST_LOG_TRIVIAL(debug) << "Computed " << _interestPoints.size() << " keypoints";

    BOOST_LOG_TRIVIAL(info) << "Leaving _computeKeypoints_";
}


/**
 * This function computes matches for a specific keypoint.
 * It calls _similarityVector_ in order to compute the similarity
 * vector for keypoint _i_ then adds to allMatches the matched keypoints.
 *
 * @param i     The index of the source keypoint in _keypoints.
 */
void copyMoveDetector::computeMatch(int i) {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _computeMatch_";

    BOOST_LOG_TRIVIAL(trace) << "Computing match for keypoint n°" << i;
    map<double, InterestPoint>&& similarity = _interestPoints.similarityAngle(_interestPoints[i]);

    map<double, InterestPoint>::iterator it1;
    map<double, InterestPoint>::iterator it2;
    vector<InterestPoint> matches;

    /*
     * Here, the similarity vector should be sorted by ascending order.
     * Thus, we iterate over it with two iterators and we compare the
     * euclidean distances. Mathematically, we have a vector
     *              D = {d1, ..., d_n}
     * and we check for i in [1, n-1] :
     *              di / di+1 < T
     */
    for (it1 = similarity.begin(),
         it2 = next(it1);
         it1 != similarity.end() && it2 != similarity.end();
         it1++, it2++) {
        double distance1 = it1->first;
        double distance2 = it2->first;
        if (distance1 / distance2 < 0.5 && it1->second.getAngleIdx() != i) {
            InterestPoint& possibleMatch = it1->second;
            int j = possibleMatch.getAngleIdx();
            /*
             * First we check that the match hasn't already been computed.
             */
            if (find(_allMatches[j].begin(), _allMatches[j].end(), _interestPoints[i]) == _allMatches[j].end())
                matches.push_back(possibleMatch);
            break;
        }
        else
            break;
    }

    _allMatches[i] = matches;

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _computeMatch_";
}

/**
 * This function is a friend of copyMoveDetector.
 * It isn't part of the class because it is going to be called by
 * a thread and a thread can't call a member function.
 *
 * It computes all matches for keypoints in range [start, end[.
 *
 * @param   detector    The detector we want to compute a match.
 * @param   start       The start index.
 * @param   end         The end index.
 */
void defals::runMatches(copyMoveDetector& detector, int start, int end) {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _runMatches_";

    BOOST_LOG_TRIVIAL(debug) << "Starting computation of matches [" << start << ", " << start + end << "[";
    for (int i = start; i < start + end; i++) {
        detector.computeMatch(i);
    }

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _runMatches_";
}

/**
 * @copydoc defals::runMatches(copyMoveDetector&,int,int)
 */
void defals::runBetterMatches(copyMoveDetector &detector, int start, int end) {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _runBetterMatches_";

    BOOST_LOG_TRIVIAL(debug) << "Starting computation of matches [" << start << ", " << start + end << "[";
    for (int i = start; i < start + end; i++) {
        detector.computeBetterMatch(i);
    }

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _runBetterMatches_";
}

string infoKeypoint(const InterestPoint& pt) {
    string result;
    result += "\tPosition = (" + to_string(pt.pt.x) + ", " + to_string(pt.pt.y) + ")\n";
    result += "\tAngle = " + to_string(pt.angle) + "\n";
    result += "\tIndex in angles : " + to_string(pt.getAngleIdx()) + "\n";
    result += "\tIndex in norms: " + to_string(pt.getNormIdx()) + "\n";

    return result;
}

void copyMoveDetector::computeBetterMatch(int i) {
    BOOST_LOG_TRIVIAL(trace) << "--> Entering _computeBetterMatch_";

    string logString;

    InterestPoint pi = _interestPoints[i];
    logString += "Looking for matches for InterestPoint:\n";
    logString += infoKeypoint(pi);

    pair<int, int>&& rangeAngle = _interestPoints.getRangeAngle(pi);
    int minIdxAngle = rangeAngle.first;
    int maxIdxAngle = rangeAngle.second;

    logString += "Angle window size: " + to_string(maxIdxAngle - minIdxAngle) + " points\n";

    map<double, InterestPoint>&& similarityAngles = _interestPoints.similarityAngle(pi, minIdxAngle, maxIdxAngle);

    vector<InterestPoint> matches;
    if (!similarityAngles.empty()) {
        map<double, InterestPoint>::iterator it1, it2;
        for (it1 = similarityAngles.begin(),
                     it2 = next(it1);
             it1 != similarityAngles.end() && it2 != similarityAngles.end();
             it1++, it2++) {
            double distance1 = it1->first;
            double distance2 = it2->first;
            if (distance1 / distance2 < 0.5 && it1->second.getAngleIdx() != pi.getAngleIdx()) {
                InterestPoint &pj = it1->second;
                int j = pj.getAngleIdx();

                if (find(_allMatches[j].begin(), _allMatches[j].end(), _interestPoints[i]) == _allMatches[j].end())
                    matches.push_back(pj);
            } else
                break;
        }

        logString += "InterestPoint has been matched with " + to_string(matches.size()) + " points\n";

        _allMatches[i] = matches;
    }

    if (!matches.empty())
        BOOST_LOG_TRIVIAL(trace) << logString;
    BOOST_LOG_TRIVIAL(trace) << "<-- Leaving _computeBetterMatch_";
}

/**
 * This function splits the keypoints indexes in descriptorEquals parts
 * for each thread. Then it creates a thread and makes it process
 * its own slice of keypoints.
 * It waits for all threads to finish running before exiting.
 */
void copyMoveDetector::computeMatches() {
    int nbMatches = _interestPoints.size();
    _allMatches = vector<vector<InterestPoint>>(nbMatches);

    const int nbThreads = _options.jobs;
    int matchesByThread = nbMatches / nbThreads;
    vector<thread> threads;
    for (int noThread = 0; noThread < nbThreads; noThread++) {
        int start = noThread * matchesByThread;
        thread t(runMatches, ref(*this), start, matchesByThread);
        threads.push_back(move(t));
    }

    for (auto& t : threads)
        t.join();

}

void copyMoveDetector::computeBetterMatches() {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _computeBetterMatches_";

    int nbMatches = _interestPoints.size();
    _allMatches = vector<vector<InterestPoint>>(nbMatches);
    BOOST_LOG_TRIVIAL(debug) << "Looking for " << nbMatches << " matches";

    const int nbThreads = _options.jobs;
    int matchesByThreads = nbMatches / nbThreads;
    BOOST_LOG_TRIVIAL(debug) << "Launching search with " << nbThreads << " threads (" << matchesByThreads << " points by thread)";
    vector<thread> threads;
    for (int noThread = 0; noThread < nbThreads; noThread++) {
        int start = noThread * matchesByThreads;
        BOOST_LOG_TRIVIAL(debug) << "Launching thread " << noThread << " on interval [" << start << ", " << start + matchesByThreads << "]";
        thread t(runBetterMatches, ref(*this), start, matchesByThreads);
        threads.push_back(move(t));
    }

    BOOST_LOG_TRIVIAL(debug) << "Waiting for threads to finish";
    for (auto& t : threads)
        t.join();

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _computeBetterMatches_";
}

/**
 *
 * @tparam T
 * @param p1
 * @param p2
 * @return
 */
template<typename T>
inline bool lex(const T& p1, const T& p2) {
    return p1.x < p2.x || (p1.x == p2.x && p1.y <= p2.y);
}

/**
 * This function creates all the lines by associating each keypoint to its matched
 * keypoints. It doesn't add the lines whose length is shorter than _minLength_.
 *
 * @param   minLength   Minimum length required for accepting a line.
 */
void copyMoveDetector::computeLines() {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _computeLines_";

    for (size_t i = 0; i < _allMatches.size(); i++) {
        vector<InterestPoint> matches = _allMatches[i];
        InterestPoint origin = _interestPoints.get(i);
        for (const auto& pt : matches) {
            Line droite(origin, pt);
            BOOST_LOG_TRIVIAL(trace) << "Created line [(" <<
                                     origin.pt.x << ", " << origin.pt.y << "), (" <<
                                     pt.pt.x << ", " << pt.pt.y << ") of length " << droite.length();
            if (lex(pt.pt, origin.pt))
                droite.invert();
            if (droite.length() >= _options.length)
                _lines.push_back(droite);
        }
    }

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _computeLines_";
}

/**
 * This function computes all clusters using DBSCAN algorithm.
 *
 * @param threshold     Minimal number of lines in a cluster.
 */
void copyMoveDetector::computeClusters() {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _computeClusters_";

    vector<ClusteredLine> lines(_lines.begin(), _lines.end());

    bool ok = false;
    while (!ok && _options.dbscan_minPts >= 2) {
        // Parameters for GRIP : minPts = 4 ; eps = 1000
        DBSCAN scanner(_options.dbscan_minPts, _options.dbscan_epsilon, lines,
                       _image.rows, _image.cols,
                       _options.dbscan_wx, _options.dbscan_wy, _options.dbscan_wtheta);
        //DBSCAN scanner(4, 9000, lines, _image.rows, _image.cols);

        BOOST_LOG_TRIVIAL(debug) << "Starting scanner with parameters minPts = " << _options.dbscan_minPts <<
                                    " and epsilon = " << _options.dbscan_epsilon;

        vector<ClusteredLine> clusteredLines = scanner.run();

        int nbClusters = 0;
        for (const auto &line : clusteredLines) {
            int id = line.getId();
            if (id > 0) {
                if (id > nbClusters)
                    nbClusters = id;
            }
        }

        vector<Cluster> clusters(nbClusters);
        for (const auto &line : clusteredLines) {
            int id = line.getId();
            if (id > 0)  // Hey don't forget indices start at 0
                clusters[id - 1].push_back(line);
            else
                _outliers.push_back(line);
        }

        for (int i = 0; i < nbClusters; i++) {
            if (clusters[i].size() >= 3)
                _clusters.push_back(clusters[i]);
        }

        if (!_clusters.empty()) {
            BOOST_LOG_TRIVIAL(debug) << "Computed " << _clusters.size() << " clusters";
            ok = true;
        }
        else {
            BOOST_LOG_TRIVIAL(debug) << "No cluster found";
            _options.dbscan_minPts /= 2;
        }
    }

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _computeClusters_";
}

/**
 * TODO:
 * Reference: Mayer O Stamm Forensic similarity for digital images IEEE TIFS
 */

void copyMoveDetector::computeHull() {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _computeHull_";

    for (const auto& cluster : _clusters) {
        vector<tuple<Point, InterestPoint, Line>> starts, ends;
        for (const auto& line : cluster) {
            starts.emplace_back(line.getPoint1().pt, line.getPoint1(), line);
            ends.emplace_back(line.getPoint2().pt, line.getPoint2(), line);
        }

        vector<Point> actualStarts, actualEnds;
        for (const auto& p : starts)
            actualStarts.push_back(get<0>(p));
        for (const auto& p : ends)
            actualEnds.push_back(get<0>(p));

        vector<int> hullStarts, hullEnds;
        convexHull(actualStarts, hullStarts);
        convexHull(actualEnds, hullEnds);

        vector<pair<InterestPoint, Line>> ptsStarts, ptsEnds;

        for (const auto& i : hullStarts)
            ptsStarts.emplace_back(get<1>(starts[i]), get<2>(starts[i]));
        _hulls.emplace_back(ptsStarts);

        BOOST_LOG_TRIVIAL(debug) << "Initial hull contains " << ptsStarts.size() << " points";

        for (const auto& i : hullEnds)
            ptsEnds.emplace_back(get<1>(ends[i]), get<2>(ends[i]));
        _hulls.emplace_back(ptsEnds);

        BOOST_LOG_TRIVIAL(debug) << "Matching hull contains " << ptsEnds.size() << " points";
    }

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _computeHull_";
}

void copyMoveDetector::computeMask(int kernelSize) {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _computeMask_";

    _computedMask = Mat::zeros(_image.size(), CV_8UC1);
    Scalar white(0xFF);

    vector<vector<Point>> hulls;
    for (const auto& hull : _hulls) {
        vector<Point> pts;
        for (const auto& pt : hull) {
            pts.push_back(pt.first.pt);
        }
        hulls.push_back(pts);
    }
    drawContours(_computedMask, hulls, -1, white, FILLED);

    if (_options.before_dilation)
        save(_computedMask, _options.rawName + "_6mask_before_dilation.png");

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _computeMask_";
}

inline const InterestPoint& getMatch(const InterestPoint& pt, const Line& line) {
    if (pt == line.getPoint1())
        return line.getPoint2();
    else if (pt == line.getPoint2())
        return line.getPoint1();

    BOOST_LOG_TRIVIAL(error) << "Point not in line";
    exit(1);
}

inline Vec3f RGBtoYCbCr(const Vec3b& BGR) {
    uchar B = BGR[0];
    uchar G = BGR[1];
    uchar R = BGR[2];

    double Y = 0.299 * R + 0.587 * G + 0.114 * B;
    double Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128;
    double Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128;

    return Vec3f(Y, Cb, Cr);
}

/**
 * This function computes all neighbours in an elliptic area around a point.
 *
 * @param pt        The point at the center of the ellipse.
 * @param maxX      The maximum of the X axis.
 * @param maxY      The maximum of the Y axis.
 * @param ksize     The size of the square the ellipse will be fitting into.
 *                  Must be an odd number.
 *
 * @return  A list of points which are in the elliptic area.
 */
vector<Point> neighbours(const Point& pt,
                         int maxX,
                         int maxY,
                         [[odd]] const Size& ksize = Size(3, 3)) {
    if (!ksize.width % 2 || !ksize.height % 2) {
        BOOST_LOG_TRIVIAL(error) << "kernel size must be odd";
        exit(1);
    }

    vector<Point> coordinates;

    int infX = pt.x - ksize.width / 2;
    if (infX < 0)
        infX = 0;

    int infY = pt.y - ksize.height / 2;
    if (infY < 0)
        infY = 0;

    int supX = pt.x + ksize.width / 2;
    if (supX >= maxX)
        supX = maxX - 1;

    int supY = pt.y + ksize.height / 2;
    if (supY >= maxY)
        supY = maxY - 1;

    for (int x = infX; x <= supX; x++) {
        for (int y = infY; y <= supY; y++) {
            Point other(x, y);
            double distance = sqrt((pt.x - other.x) * (pt.x - other.x) + (pt.y - other.y) * (pt.y - other.y));
            if (distance <= ksize.width / 2)
                coordinates.emplace_back(other);
        }
    }
    return coordinates;
}

/**
 * Given a set of points representing a convex hull, computes all the intermediate
 * pixels between each point of the hull.
 *
 * @param hull      The hull.
 *
 * @return      A set of points representing the pixels in the border of the hull.
 */
vector<pair<Point, Point>> copyMoveDetector::borderOfHull(const vector<pair<Point, Point>>& hull) const {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _borderOfHull_ (non InterestPoint)";

    vector<pair<Point, Point>> pts;

    auto it1 = hull.begin();
    auto it2 = next(it1);
    for (; it1 != hull.end() && it2 != hull.end();
           it1++, it2++) {
        const Point& one = it1->first;
        const Point& matchOne = it1->second;
        const Point& two = it2->first;
        const Point& matchTwo = it2->second;

        pts.emplace_back(one, matchOne);

        LineIterator intermediaires(_image, one, two, 8, true);
        LineIterator matches(_image, matchOne, matchTwo, 8, true);

        size_t smallest = intermediaires.count < matches.count ? intermediaires.count : matches.count;

        for (size_t i = 0; i < smallest; i++, intermediaires++, matches++) {
            pts.emplace_back(intermediaires.pos(), matches.pos());
        }

        pts.emplace_back(two, matchTwo);
    }
    const Point& one = pts[pts.size() - 1].first;
    const Point& matchOne = pts[pts.size() - 1].second;
    const Point& two = pts[0].first;
    const Point& matchTwo = pts[0].second;

    LineIterator intermediaires(_image, one, two, 8, true);
    LineIterator matches(_image, matchOne, matchTwo, 8, true);

    size_t smallest = intermediaires.count < matches.count ? intermediaires.count : matches.count;

    BOOST_LOG_TRIVIAL(debug) << "Hull border contains " << smallest << " points";

    for (size_t i = 0; i < smallest; i++, intermediaires++, matches++)
        pts.emplace_back(intermediaires.pos(), matches.pos());


    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _borderOfHull_ (non InterestPoint)";
    return pts;
}

/**
 * Same as copyMoveDetector::borderOfHull(const vector<pair<Point, Point>>&)
 * but acts on a hull computed by the algorithm.
 *
 * @param i     The hull index in _hulls.
 *
 * @return      A vector of pixels located in the border of the hull.
 */
vector<pair<Point, Point>> copyMoveDetector::borderOfHull(int i) const {
    BOOST_LOG_TRIVIAL(debug) << "--> Entering _borderOfHull_ (InterestPoint)";

    vector<pair<Point, Point>> pts;

    auto it1 = _hulls[i].begin();
    auto it2 = next(it1);
    for (; it1 != _hulls[i].end() && it2 != _hulls[i].end();
           it1++, it2++) {
        const InterestPoint& one = it1->first;
        const InterestPoint& matchOne = getMatch(one, it1->second);
        const InterestPoint& two = it2->first;
        const InterestPoint& matchTwo = getMatch(two, it2->second);

        pts.emplace_back(one.pt, matchOne.pt);

        LineIterator intermediaires(_image, one.pt, two.pt, 8, true);
        LineIterator matches(_image, matchOne.pt, matchTwo.pt, 8, true);

        size_t smallest = intermediaires.count < matches.count ? intermediaires.count : matches.count;

        for (size_t i = 0; i < smallest; i++, intermediaires++, matches++) {
            pts.emplace_back(intermediaires.pos(), matches.pos());
        }

        pts.emplace_back(two.pt, matchTwo.pt);
    }
    const Point& one = pts[pts.size() - 1].first;
    const Point& matchOne = pts[pts.size() - 1].second;
    const Point& two = pts[0].first;
    const Point& matchTwo = pts[0].second;

    LineIterator intermediaires(_image, one, two, 8, true);
    LineIterator matches(_image, matchOne, matchTwo, 8, true);

    size_t smallest = intermediaires.count < matches.count ? intermediaires.count : matches.count;

    BOOST_LOG_TRIVIAL(debug) << "Hull border contains " << smallest << " points";

    for (size_t j = 0; j < smallest; j++, intermediaires++, matches++)
        pts.emplace_back(intermediaires.pos(), matches.pos());

    BOOST_LOG_TRIVIAL(debug) << "<-- Leaving _borderOfHull_ (InterestPoint)";
    return pts;
}

double copyMoveDetector::computePSNR(int i) const {
    vector<vector<Point>> hulls;

    vector<Point> firstHull;
    for (const auto& pt : _hulls[i])
        firstHull.emplace_back(pt.first.pt);
    hulls.emplace_back(firstHull);

    vector<Point> secondHull;
    for (const auto& pt : _hulls[i + 1])
        secondHull.emplace_back(pt.first.pt);
    hulls.emplace_back(secondHull);


    Mat hull1 = Mat::zeros(_image.size(), CV_8UC1);
    drawContours(hull1, hulls, 0, Scalar(0xFF), FILLED);
    Mat hull2 = Mat::zeros(_image.size(), CV_8UC1);
    drawContours(hull2, hulls, 1, Scalar(0xFF), FILLED);

    vector<Point> firstIndices, secondIndices;
    for (int x = 0; x < _image.cols; x++) {
        for (int y = 0; y < _image.rows; y++) {
            uchar intensity1 = hull1.at<uchar>(y, x);
            uchar intensity2 = hull2.at<uchar>(y, x);

            if (intensity1 == 0xFF)
                firstIndices.emplace_back(x, y);
            if (intensity2 == 0xFF)
                secondIndices.emplace_back(x, y);
        }
    }

    size_t smallest = firstIndices.size() < secondIndices.size() ? firstIndices.size() : secondIndices.size();

    double EQM = 0;
    for (size_t j = 0; j < smallest; j++) {
        Point& coord1 = firstIndices[j];
        Vec3b BGR1 = _image.at<Vec3b>(coord1);
        uchar&& Y1 = RGBtoYCbCr(BGR1)[0];

        Point& coord2 = secondIndices[j];
        Vec3b BGR2 = _image.at<Vec3b>(coord2);
        uchar&& Y2 = RGBtoYCbCr(BGR2)[0];

        EQM += (Y1 - Y2) * (Y1 - Y2);
    }
    EQM /= smallest;

    double PSNR = 10 * log(255 * 255 / EQM);

    BOOST_LOG_TRIVIAL(info) << "PSNR threshold computed to: " << PSNR;

    return PSNR;
}

void copyMoveDetector::extendMask() {
    if (!_hulls.empty()) {
        int compteur = 7;
        _extendedMask = _computedMask.clone();
        for (size_t i = 0; i < _hulls.size(); i += 2) {
            //double PSNR_threshold = computePSNR(i) + 50;

            vector<pair<Point, Point>>&& border = borderOfHull(i);

            Size ksize(13, 13);

            bool finished = false;
            int j = 0;
            while (!finished && j < 20) {
                finished = true;
                vector<pair<Point, Point>> newPoints(border);

                for (const auto &match : border) {
                    const Point &pt1 = match.first;
                    const Point &pt2 = match.second;

                    bool atBorder = false;
                    vector<pair<Point, Point>> &&addedPoints = checkEQM(pt1, pt2, atBorder, ksize, _options.PSNR);
                    /*
                    if (addedPoints.empty()) {
                        ksize.width /= 2;
                        ksize.height /= 2;

                        addedPoints = checkEQM(pt1, pt2, atBorder, ksize, _options.PSNR);
                    }*/
                    if (!atBorder)
                        finished = false;
                    BOOST_LOG_TRIVIAL(debug) << "Computed EQM with " << ksize.width
                                             << "x" << ksize.height << " window.";
                    ksize.width = 13;
                    ksize.height = 13;

                    newPoints.insert(newPoints.end(), addedPoints.begin(), addedPoints.end());
                }

                vector<Point> starts;
                for (const auto &match : newPoints)
                    starts.emplace_back(match.first);

                vector<int> newHull;
                convexHull(starts, newHull);

                vector<pair<Point, Point>> newBorder;
                for (const auto &idx : newHull)
                    newBorder.emplace_back(newPoints[idx]);

                border = borderOfHull(newBorder);

                if (_options.stepByStep_expansion) {
                    save(_extendedMask, _options.rawName + "_" + to_string(++compteur) + "mask_extended_" + to_string(i)
                                        + '-' + to_string(j) + ".jpg");
                }
                j++;
            }
        }
    }
}

vector<pair<Point, Point>> copyMoveDetector::checkEQM(const cv::Point &pt1,
                                const cv::Point &pt2,
                                bool& border,
                                const cv::Size &ksize,
                                const double PSNR_threshold) {
    vector<Point>&& kernelOne = neighbours(pt1,
                                           _image.cols, _image.rows,
                                           ksize);
    vector<Point>&& kernelTwo = neighbours(pt2,
                                           _image.cols, _image.rows,
                                           ksize);

    vector<Point>& actualOne = kernelOne.size() <= kernelTwo.size() ? kernelOne : kernelTwo;
    vector<Point>& actualTwo = kernelOne == actualOne ? kernelTwo : kernelOne;

    double EQM = 0;
    for (size_t i = 0; i < actualOne.size(); i++) {
        const Point& one = actualOne[i];
        const Point& two = actualTwo[i];

        Vec3b BGRone = _image.at<Vec3b>(one);
        double Yone = RGBtoYCbCr(BGRone)[0];

        Vec3b BGRtwo = _image.at<Vec3b>(two);
        double Ytwo = RGBtoYCbCr(BGRtwo)[0];

        EQM += (Yone - Ytwo) * (Yone - Ytwo);
    }
    EQM /= actualOne.size();


    vector<pair<Point, Point>> addedPoints;

    double PSNR = 10 * log(255 * 255 / EQM);
    BOOST_LOG_TRIVIAL(debug) << "EQM = " << EQM
                             << ". PSNR = " << PSNR;

    if (PSNR >= PSNR_threshold) {
        border = false;
        for (size_t i = 0; i < actualOne.size(); i++) {
            const Point& one = actualOne[i];
            const Point& two = actualTwo[i];

            _extendedMask.at<uchar>(one) = 0xFF;
            _extendedMask.at<uchar>(two) = 0xFF;

            addedPoints.emplace_back(one, two);
        }
    }
    else {
        border = true;
        for (size_t i = 0; i < actualOne.size(); i++) {
            const Point& one = actualOne[i];
            const Point& two = actualTwo[i];

            Vec3b BGRone = _image.at<Vec3b>(one);
            Vec3b BGRtwo = _image.at<Vec3b>(two);

            double Yone = RGBtoYCbCr(BGRone)[0];
            double Ytwo = RGBtoYCbCr(BGRtwo)[0];

            if (abs(Yone - Ytwo) <= 0) {
                _extendedMask.at<uchar>(one) = 0xFF;
                _extendedMask.at<uchar>(two) = 0xFF;

                addedPoints.emplace_back(one, two);
            }
        }
    }

    return addedPoints;
}

double copyMoveDetector::computeDice() const {
    if (_mask.empty())
        return -1;

    if (_extendedMask.empty())
        return -1;

    int X = 0, Y = 0, XinterY = 0;

    for (int x = 0; x < _image.cols; x++) {
        for (int y = 0; y < _image.rows; y++) {
            uchar original = _mask.at<uchar>(y, x);
            uchar computed = _extendedMask.at<uchar>(y, x);

            if (original == 0xFF)
                X++;
            if (computed == 0xFF)
                Y++;
            if (original == 0xFF && computed == 0xFF)
                XinterY++;
        }
    }

    BOOST_LOG_TRIVIAL(info) << "Jaccard = " << (double) XinterY / (double) (X + Y - XinterY);

    return 2 * (double) XinterY / (double) (X + Y);
}

void copyMoveDetector::computeFScore(double& precision, double& recall, double& F1) const {
    if (_mask.empty() || _extendedMask.empty()) {
        precision = -1;
        recall = -1;
        F1 = -1;
        return;
    }

    double Ncf = 0.0, Nff = 0.0, Nfo = 0.0;

    for (int x = 0; x < _image.cols; x++) {
        for (int y = 0; y < _image.rows; y++) {
            uchar original = _mask.at<uchar>(y, x);
            uchar computed = _extendedMask.at<uchar>(y, x);

            if (original == 0xFF && computed == 0xFF)
                Ncf++;
            else if (original == 0xFF)
                Nfo++;
            else if (computed == 0xFF)
                Nff++;
        }
    }

    precision = Ncf / (Ncf + Nff);
    recall = Ncf / (Ncf + Nfo);
    F1 = 2 * precision * recall / (precision + recall);
}


/**
 * +=================+
 * | RANDOM SEGMENTS |
 * +=================+
 *
 * Those functions should not be useful but I keep them
 * just in case.
 */

Point2f pointFrom4D(int x1, int y1, double theta, double l) {
    double sinTheta = sin(theta), cosTheta = cos(theta);
    double sin2Theta = sinTheta * sinTheta, cos2Theta = cosTheta * cosTheta;
    double rho = x1 * cosTheta + y1 * sinTheta;

    double A = l * l - x1 * x1 - y1 * y1 + 2 * y1 * rho / sinTheta - rho * rho / sin2Theta;
    double B = cos2Theta / sin2Theta + 1;
    double C = -2 * x1 + 2 * y1 * cosTheta / sinTheta - 2 * rho * cosTheta / sin2Theta;

    double Delta = C * C + 4 * A * B;

    double x21 = (-C - sqrt(Delta)) / (2 * B);
    double x22 = (-C + sqrt(Delta)) / (2 * B);

    double x2 = x1 < x21 ? x21 : x22;
    double y2 = (rho - x2 * cosTheta) / sinTheta;

    return Point2f(x2, y2);
}

Line genInitialLine(int maxWidth, int maxHeight) {
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> distX(0, maxWidth);
    uniform_int_distribution<int> distY(0, maxHeight);
    uniform_real_distribution<double> distTheta(-M_PI, M_PI);

    uniform_real_distribution<double> distLx(10 * sqrt(2), maxWidth / 2);
    uniform_real_distribution<double> distLy(10 * sqrt(2), maxHeight / 2);

    int x1 = distX(mt), y1 = distY(mt);
    double theta = distTheta(mt);
    double lx = distLx(mt), ly = distLy(mt);

    while (x1 + lx > maxWidth || x1 - lx < 0) {
        double b = distLx.b() / 2;
        if (b < 10)
            return Line(Point2f(0, 0), Point2f(0, 0));

        distLx = uniform_real_distribution<double>(10, b);
        lx = distLx(mt);
    }

    /*
    cout << "maxWidth = " << maxWidth << endl;
    cout << "x1 + lx = " << x1 + lx << "\tx1 - lx = " << x1 - lx << endl;
     */

    while (y1 + ly > maxHeight || y1 - ly < 0) {
        double b = distLy.b() / 2;
        if (b < 10)
            return Line(Point2f(0, 0), Point2f(0, 0));

        distLy = uniform_real_distribution<double>(10, b);
        ly = distLy(mt);
    }
    /*
    cout << "maxHeight = " << maxHeight << endl;
    cout << "y1 + ly = " << y1 + ly << "\ty1 - ly = " << y1 - ly << endl;
     */

    double l = sqrt(lx * lx + ly * ly);

    Point2f&& other = pointFrom4D(x1, y1, theta, l);
    return Line(Point2f(x1, y1), other);
}

Line genParallelLine(int maxWidth, int maxHeight, const Line& initial) {
    random_device rd;
    mt19937 mt(rd());

    double x0 = initial.getPoint1().pt.x;
    double y0 = initial.getPoint1().pt.y;
    double theta0 = initial.getTheta();
    double l0 = initial.length();

    uniform_int_distribution<int> distX(-0.05 * maxWidth, 0.05 * maxWidth);
    uniform_int_distribution<int> distY(-0.05 * maxHeight, 0.05 * maxHeight);
    uniform_real_distribution<double> distTheta(-M_PI / 12, M_PI / 12);
    uniform_real_distribution<double> distL(-0.1 * l0, 0.1 * l0);

    int x1 = x0 + distX(mt), y1 = y0 + distY(mt);
    double theta1 = theta0 + distTheta(mt);
    double l1 = l0 + distL(mt);

    Point2f&& extremite = pointFrom4D(x1, y1, theta1, l1);
    return Line(Point2f(x1, y1), extremite);
}

void copyMoveDetector::randomLines() {
    int maxWidth = 0.9 * _image.cols;
    int maxHeight = 0.9 * _image.rows;

    Line origin(Point2f(0, 0), Point2f(0, 0));
    for (int i = 0; i < 3; i++) {
        Line&& line = genInitialLine(maxWidth, maxHeight);
        if (line != origin) {
            _lines.emplace_back(line);
            for (int j = 0; j < 100; j++) {
                Line&& parallel = genParallelLine(maxWidth, maxHeight, line);
                if (parallel != origin)
                    _lines.emplace_back(parallel);
            }
        }
    }

    for (int i = 0; i < 50; i++) {
        Line&& line = genInitialLine(maxWidth, maxHeight);
        if (line != origin)
            _lines.emplace_back(line);
    }

}

void copyMoveDetector::conclude() const {

}

