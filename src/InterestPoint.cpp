#include "../include/InterestPoint.hpp"


using namespace std;
using namespace cv;
using namespace defals;

/**
 * Constructs an InterestPoint from a KeyPoint. The angle and norm indices are
 * set to -1 because they don't belong to any of the two vectors yet.
 */
InterestPoint::InterestPoint() : KeyPoint(), _angleIdx(-1), _normIdx(-1) {
}

/**
 * Constructs an InterestPoint from a Point2f. This constructor is useful when we
 * want to handle a simple point -- for instance in Line class -- without loosing
 * its KeyPoint characteristics.
 *
 * @param point     The point from which the InterestPoint is constructed.
 */
InterestPoint::InterestPoint(const Point2f &point) : _angleIdx(-1), _normIdx(-1){
    this->pt = point;
}

/**
 * Constructs an InterestPoint from a KeyPoint and its descriptor.
 *
 * @param keypoint      The keypoint.
 * @param descriptor    The keypoint's descriptor: 1x64 or 1x128 float matrix.
 */
InterestPoint::InterestPoint(const KeyPoint& keypoint, const Mat& descriptor) : KeyPoint(keypoint), _angleIdx(-1), _normIdx(-1) {
    _descriptor = descriptor;
}

/**
 * Equality condition in terms of descriptors' distance.
 *
 * @param other     Another InterestPoint.
 *
 * @return  True if _this_ is equal to _other_ i.e. their descriptors'
 *          euclidean distance is zero.
 */
bool InterestPoint::descriptorEquals(const InterestPoint &other) const {
    return norm(_descriptor, other._descriptor, NORM_L2) == 0;
}

/**
 * Lower condition in terms of descriptors' norms.
 *
 * @param other     Another InterestPoint.
 *
 * @return  True if _this_ is lower in norm than _other_ i.e. the difference of the norms
 *          of their respective descriptors is negative.
 */
bool InterestPoint::descriptorLower(const InterestPoint &other) const {
    return norm(_descriptor, NORM_L2) - norm(other._descriptor, NORM_L2) < 0;
}

/**
 * Lower condition in terms of angles.
 *
 * @param other     Another InterestPoint.
 *
 * @return  True if _this_ is lower in angle than _other_ i.e. the difference of their angles
 *          is negative.
 */
bool InterestPoint::angleLower(const InterestPoint &other) const {
    return angle - other.angle < 0;
}

/**
 * Getter for __descriptor_.
 *
 * @return  The keypoint's descriptor.
 */
const cv::Mat& InterestPoint::getDescriptor() const {
    return _descriptor;
}

/**
 * Prints keypoint in format (x, y).
 *
 * @return  A string formatted as (x, y).
 */
std::string InterestPoint::printKeypoint() const {
    return "(" + to_string(pt.x) + ", " + to_string(pt.y) + ")";
}

/**
 * Equals condition in terms of angles.
 *
 * @param other     Another InterestPoint.
 *
 * @return  True if _this_ and _other_ have the same orientation, false otherwise.
 */
bool InterestPoint::angleEquals(const InterestPoint &other) const {
    return angle == other.angle;
}

int InterestPoint::getAngleIdx() const {
    return _angleIdx;
}

void InterestPoint::setAngleIdx(int angleIdx) {
    _angleIdx = angleIdx;
}

int InterestPoint::getNormIdx() const {
    return _normIdx;
}

void InterestPoint::setNormIdx(int normIdx) {
    _normIdx = normIdx;
}

