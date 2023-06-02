//
// Created by rubytox on 06/07/2020.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "InterestPoint.hpp"

namespace defals {
    /**
     * This class is a wrapper for an InterestPoint vector.
     *
     * Given a cv::KeyPoint vector, it handles two vectors :
     * - a vector of pointers to the keypoints, sorted by their angle value ;
     * - a vector of pointers to the keypoints, sorted by the norm of their descriptor value.
     * It is important to note that we don't duplicate points: each vector contains the exact same points
     * that have been passed to the constructor, but sorted in a different order.
     *
     * Note: unless specified otherwise, the i-th keypoint is the i-th keypoint of the angles vector.
     */
    class InterestPoints {
    public:
        /*
         * +================+
         * |  CONSTRUCTORS  |
         * +================+
         */
        InterestPoints() = default;

        InterestPoints(const std::vector <cv::KeyPoint> &keypoints,
                       const cv::Mat &descriptors,
                       double angleThreshold,
                       double normThreshold);

        /*
         * +=============+
         * |  ITERATORS  |
         * +=============+
         */
        std::vector<std::shared_ptr<InterestPoint>>::iterator begin();
        std::vector<std::shared_ptr<InterestPoint>>::iterator end();

        std::vector<std::shared_ptr<InterestPoint>>::const_iterator begin() const;
        std::vector<std::shared_ptr<InterestPoint>>::const_iterator end() const;

        /*
         * +===================+
         * |  GETTERS/SETTERS  |
         * +===================+
         */
        const cv::Mat &getDescriptor(int i) const;
        cv::Mat getDescriptors() const;

        std::vector <cv::KeyPoint> asKeyPoints() const;

        const cv::Point2f &pt(int i) const;
        const InterestPoint &get(int i) const;
        const InterestPoint &operator[](int i) const;

        int size() const;

        /*
         * +=============+
         * |  ALGORITHM  |
         * +=============+
         */

        std::pair<int, int> getRangeAngle(const InterestPoint &pt) const;
        std::pair<int, int> getRangeNorm(const InterestPoint &pt) const;
        std::pair<int, int> getRelativeRangeNorm(const InterestPoint& center, const InterestPoint& end) const;

        std::map<double, InterestPoint> similarityAngle(const InterestPoint& pt, int minIdx = 0, int maxIdx = -1) const;
        std::map<double, InterestPoint> similarityNorm(const InterestPoint& pt, int minIdx = 0, int maxIdx = -1) const;

        void sort();


    private:
        std::pair<int, int> getRange(const std::vector<std::shared_ptr<InterestPoint>> &pts,
                                     int i,
                                     double (comp)(const InterestPoint &, const InterestPoint &),
                                     double threshold) const;

        /**  Pointers to the keypoints sorted by angles  */
        std::vector<std::shared_ptr<InterestPoint>> _pointsSortedAngle;
        /**  Pointers to the keypoints sorted by norms of their descriptors */
        std::vector<std::shared_ptr<InterestPoint>> _pointsSortedNorm;
        /**  The threshold for the computation of the window in the angles vector  */
        double _angleThreshold;
        /**  The threshold for the computation of the window in the norms vector  */
        double _normThreshold;
    };
}
