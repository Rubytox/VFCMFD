//
// Created by rubytox on 06/07/2020.
//

#include "../include/InterestPoints.hpp"

using namespace std;
using namespace cv;
using namespace defals;

/**
 * Constructs a list of InterestPoint from keypoints and their descriptors.
 *
 * The two vectors are initialized with pointers to the InterestPoint objects.
 *
 * @param keypoints     A vector of keypoints.
 * @param descriptors   The descriptors of the keypoints such as line i is the i-th keypoint's descriptor.
 */
InterestPoints::InterestPoints(const vector<KeyPoint>& keypoints, const Mat& descriptors,
                               double angleThreshold, double normThreshold) {
    for (size_t i = 0; i < keypoints.size(); i++) {
        KeyPoint keypoint = keypoints[i];
        Mat descriptor = descriptors.row(i);
        shared_ptr<InterestPoint> point = make_shared<InterestPoint>(keypoint, descriptor);
        _pointsSortedAngle.push_back(std::ref(point));
        _pointsSortedNorm.emplace_back(std::ref(point));
    }

    _angleThreshold = angleThreshold;
    _normThreshold = normThreshold;

    sort();
}

/**
 * This function sorts the two vectors by angle and norm.
 */
void InterestPoints::sort() {
    std::sort(_pointsSortedNorm.begin(), _pointsSortedNorm.end(), [](const shared_ptr<InterestPoint>& a, const shared_ptr<InterestPoint>& b) {
        return a->descriptorLower(*b);
    });
    std::sort(_pointsSortedAngle.begin(), _pointsSortedAngle.end(), [](const shared_ptr<InterestPoint>& a, const shared_ptr<InterestPoint>& b) {
        return a->angleLower(*b);
    });

    /*
     * Once sorted, we tell each InterestPoint their position in the vectors.
     */
    for (int i = 0; i < size(); i++) {
        _pointsSortedAngle[i]->setAngleIdx(i);
        _pointsSortedNorm[i]->setNormIdx(i);
    }
}

/**
 * @return  An iterator at the start of the angles vector.
 */
vector<shared_ptr<InterestPoint>>::iterator InterestPoints::begin() {
    return _pointsSortedAngle.begin();
}

/**
 * @return  An iterator at the end of the angles vector.
 */
vector<shared_ptr<InterestPoint>>::iterator InterestPoints::end() {
    return _pointsSortedAngle.end();
}

/**
 * @return  A const iterator at the start of the angles vector.
 */
std::vector<shared_ptr<InterestPoint>>::const_iterator InterestPoints::begin() const {
    return _pointsSortedAngle.begin();
}

/**
 * @return  A const iterator at the end of the angles vector.
 */
std::vector<shared_ptr<InterestPoint>>::const_iterator InterestPoints::end() const {
    return _pointsSortedAngle.end();
}

/**
 * @return  The number of InterestPoint.
 */
int InterestPoints::size() const {
    return _pointsSortedAngle.size();
}

/**
 * Gets the descriptor of the i-th keypoint.
 *
 * @param i     The index of the InterestPoint in the angles vector.
 *
 * @return  The descriptor of the i-th keypoint.
 */
const cv::Mat& InterestPoints::getDescriptor(int i) const {
    return _pointsSortedAngle[i]->getDescriptor();
}

/**
 * Wrapper for cv::KeyPoint's pt attribute.
 *
 * @param i     The index of the InterestPoint in the angles vector.
 *
 * @return      The Point2f representing the position of the keypoint in the picture.
 */
const Point2f& InterestPoints::pt(int i) const {
    return _pointsSortedAngle[i]->pt;
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
 * Practically, this method only computes the similarity vector in a range
 * [_minIdx_, _maxIdx_] of the whole InterestPoint vector.
 *
 * @param   pt      The InterestPoint we want to compute a similarity vector of.
 * @param   minIdx  The first point we're going to compute the distance with.
 * @param   maxIdx  The last point we're going to compute the distance with.
 *
 * @return  A map associating the euclidean distance between keypoint _pt_ and all the other
 *          keypoints in a [_minIdx_, _maxIdx_] window and the index of the corresponding other keypoints.
 */
std::map<double, InterestPoint> InterestPoints::similarityAngle(const InterestPoint& pt, int minIdx, int maxIdx) const {
    if (maxIdx == -1)
        maxIdx = size() - 1;

    /*
     * This map will contain pairs <euclidean_distance, keypoint_index>
     *
     * I put euclidean distance as the key because a map is ordered by keys
     */
    map<double, InterestPoint> distances;

    Mat descriptor = _pointsSortedAngle[pt.getAngleIdx()]->getDescriptor();

    for (int j = minIdx; j <= maxIdx; j++) {
        if (pt.getAngleIdx() != j) {
            Mat other = _pointsSortedAngle[j]->getDescriptor();

            double distance = norm(descriptor, other, NORM_L2);
            //if (distance <= _normThreshold)
            distances.insert({ distance, *_pointsSortedAngle[j] });
        }
    }
    return distances;
}

/**
 * @copydoc InterestPoints::similarityAngle(const InterestPoint&,int,int) const
 */
std::map<double, InterestPoint> InterestPoints::similarityNorm(const InterestPoint& pt, int minIdx, int maxIdx) const {
    if (maxIdx == -1)
        maxIdx = size() - 1;

    /*
     * This map will contain pairs <euclidean_distance, keypoint_index>
     *
     * I put euclidean distance as the key because a map is ordered by keys
     */
    map<double, InterestPoint> distances;

    Mat descriptor = _pointsSortedNorm[pt.getNormIdx()]->getDescriptor();

    for (int j = minIdx; j <= maxIdx; j++) {
        if (pt.getNormIdx() != j) {
            Mat other = _pointsSortedNorm[j]->getDescriptor();

            if (norm(other, NORM_L2) > _normThreshold)
                continue;

            double distance = norm(descriptor, other, NORM_L2);

            distances.insert({ distance, *_pointsSortedNorm[j] });
        }
    }
    return distances;
}

/**
 * @return A matrix of the descriptors of each keypoints.
 */
Mat InterestPoints::getDescriptors() const {
    Mat descriptors;
    for (const auto& point : _pointsSortedAngle) {
        descriptors.push_back(point->getDescriptor());
    }
    return descriptors;
}

/**
 * @param i     The index of the InterestPoint in the angles vector.
 *
 * @return      The i-th keypoint.
 */
const InterestPoint &InterestPoints::get(int i) const {
    return *_pointsSortedAngle[i];
}

/**
 * Casts all the InterestPoint to cv::KeyPoint.
 *
 * @return  A vector of cv::KeyPoint corresponding to the InterestPoint.
 */
vector<KeyPoint> InterestPoints::asKeyPoints() const {
    vector<KeyPoint> points;
    for (const auto& pt : _pointsSortedAngle) {
        points.emplace_back(*pt);
    }

    return points;
}

/**
  * Overload of operator[] in order to access keypoints as in a vector.
  *
 * @param i     The index of the keypoint in angle vector.
 *
 * @return      The i-th InterestPoint.
 */
const InterestPoint &InterestPoints::operator[](int i) const {
    assert(i >= 0 && (size_t) i < _pointsSortedAngle.size());

    return *_pointsSortedAngle[i];
}

/**
 * Given an InterestPoint _pt_, computes the indices of a window
 * around _pt_ containing only points whose angle is not further
 * from _pt_'s by __angleThreshold_ degrees.
 *
 * @param pt    The InterestPoint we want a window around.
 *
 * @return  A pair of indices representing the window [_minIdx_, _maxIdx_]
 */
pair<int, int> InterestPoints::getRangeAngle(const InterestPoint& pt) const {
    return getRange(_pointsSortedAngle,
                    pt.getAngleIdx(),
                    [](const InterestPoint& a, const InterestPoint& b) {
                        double angleA = a.angle;
                        double angleB = b.angle;
                        return abs(angleA - angleB);
                    },
                    _angleThreshold);

}

/**
 * Given an InterestPoint _pt_, computes the indices of a window
 * around _pt_ containing only points whose descriptor's norm is not further
 * from _pt_'s by __normThreshold_.
 *
 * @param pt    The InterestPoint we want a window around.
 *
 * @return  A pair of indices representing the window [_minIdx_, _maxIdx_]
 */
pair<int, int> InterestPoints::getRangeNorm(const InterestPoint& pt) const {
    return getRange(_pointsSortedNorm,
                    pt.getNormIdx(),
                    [](const InterestPoint& a, const InterestPoint& b) {
                        return abs(norm(a.getDescriptor(), NORM_L2) - norm(b.getDescriptor(), NORM_L2));
                    },
                    _normThreshold);
}


/**
 * This function actually computes the [minIdx, maxIdx] window described in:
 * - InterestPoints::getRangeAngle(const InterestPoint&) const
 * - InterestPoints::getRangeNorm(const InterestPoint&) const
 *
 * @param pts           The **SORTED** vector from which the window will be selected.
 * @param i             The index of the keypoint at the center of the window in _pts_.
 * @param comp          The comparison function.
 * @param threshold     The threshold above which a point will not be selected.
 *
 * @return  A window [minIdx, maxIdx] around the _i_-th keypoint of _pts_ such as:
 *
 *          \f$ \forall j \in [\mathrm{minIdx}, \mathrm{maxIdx}], comp(pts[i], pts[j]) < \mathrm{threshold}\f$
 */
pair<int, int> InterestPoints::getRange(const vector<shared_ptr<InterestPoint>>& pts,
                                        int i,
                                        double (comp) (const InterestPoint&, const InterestPoint&),
                                        double threshold) const {
    InterestPoint pt = *pts[i];

    int minIdx = i;
    while (minIdx >= 0) {
        InterestPoint other = *pts[minIdx];

        if (comp(pt, other) >= threshold)
            break;

        minIdx--;
    }
    if (minIdx < 0)
        minIdx = 0;

    int maxIdx = i;
    while (maxIdx < size()) {
        InterestPoint other = *pts[maxIdx];

        if (comp(pt, other) >= threshold)
            break;

        maxIdx++;
    }
    if (maxIdx >= size())
        maxIdx = size() - 1;

    return make_pair(minIdx, maxIdx);
}

pair<int, int> InterestPoints::getRelativeRangeNorm(const InterestPoint& center, const InterestPoint& end) const {
    int centerIdx = center.getNormIdx();
    int endIdx = end.getNormIdx();

    int minIdx, maxIdx;
    if (endIdx < centerIdx) {
        minIdx = endIdx;
        maxIdx = 2 * centerIdx - endIdx;
    }
    else {
        maxIdx = endIdx;
        minIdx = 2 * centerIdx - endIdx;
    }
    if (minIdx < 0)
        minIdx = 0;
    if (maxIdx >= size())
        maxIdx = size() - 1;

    /*
    pair<int, int>&& before = getRangeNorm(*_pointsSortedNorm[minIdx]);
    pair<int, int>&& after = getRangeNorm(*_pointsSortedNorm[maxIdx]);

    return make_pair(before.first, after.second);
     */

    return make_pair(minIdx, maxIdx);
}
