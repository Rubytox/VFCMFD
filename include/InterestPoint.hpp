#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace defals {
    /**
     * This class acts like a wrapper for cv::KeyPoint. It associated a KeyPoint with
     * its descriptor instead of having to find it as a row of a matrix.
     *
     * The matching algorithm needs to consider two lists of InterestPoint:
     * - one ordered by the keypoint's angle ;
     * - the other ordered by the keypoint's descriptor's norm.
     * However, we need to be able to find efficiently where is one keypoint in both lists.
     * To do so, we provide an InterestPoint with two attributes standing for their index in both
     * lists.
     */
    class InterestPoint : public cv::KeyPoint {
    public:
        /*
         * +================+
         * |  CONSTRUCTORS  |
         * +================+
         */
        InterestPoint();

        InterestPoint(const cv::KeyPoint &keypoint, const cv::Mat &descriptor);

        InterestPoint(const cv::Point2f& point);

        /*
         * +===================+
         * |  GETTERS/SETTERS  |
         * +===================+
         */
        const cv::Mat &getDescriptor() const;

        int getAngleIdx() const;
        void setAngleIdx(int angleIdx);

        int getNormIdx() const;
        void setNormIdx(int normIdx);

        /*
         * +==============+
         * |  COMPARISON  |
         * +==============+
         */
        bool descriptorEquals(const InterestPoint &other) const;
        bool descriptorLower(const InterestPoint &other) const;

        bool angleEquals(const InterestPoint &other) const;
        bool angleLower(const InterestPoint& other) const;


        /*
         * +===========+
         * |  DISPLAY  |
         * +===========+
         */

        std::string printKeypoint() const;

        /**
         * This method is not defined in the source file because it's a template.
         * It just prints nicely the descriptor of the considered keypoint.
         *
         * @tparam T    Matrix data type.
         *
         * @return      A string representing the descriptor in the accurate type.
         */
        template<class T = float>
        std::string formatDescriptor() const {
            /*
             * Descriptor should be 1x64 or 1x128 matrix.
             */
            std::string result = "[";

            for (int i = 0; i < _descriptor.cols - 1; i++) {
                result += std::to_string(_descriptor.at<T>(0, i));
                result += ", ";
            }

            result += std::to_string(_descriptor.at<T>(0, _descriptor.cols - 1));

            result += "]";
            return result;
        }

    private:
        /**  A 1x64 or 1x128 vector standing for the descriptor of the keypoint */
        cv::Mat _descriptor;

        /**  The index of the keypoint in the angle-sorted list */
        int _angleIdx;
        /**  The index of the keypoint in the norm-sorted list */
        int _normIdx;
    };

    /**
     * Basically, an InterestPoint is lower than another one if its angle is lower.
     *
     * @param pt1   The first point.
     * @param pt2   The second point.
     *
     * i@return     True if pt1 < pt2, false otherwise.
     */
    inline bool operator<(const InterestPoint &pt1, const InterestPoint &pt2) {
        return pt1.angleLower(pt2);
    }

    /**
     * Two InterestPoint are equal if, and only if, they have the same orientation
     * and the same descriptor.
     *
     * @param pt1   The first point.
     * @param pt2   The second point.
     * @return      True if pt1 == pt2, false otherwise.
     */
    inline bool operator==(const InterestPoint &pt1, const InterestPoint &pt2) {
        return pt1.descriptorEquals(pt2) && pt1.angleEquals(pt2);
    }


}
