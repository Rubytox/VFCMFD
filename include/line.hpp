#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "InterestPoint.hpp"

namespace defals {

/**
 * This class provided various representations of a line :
 * - cartesian coordinates
 * - polar coordinates
 * - start point / end point representation
 */
    class Line {
    public:
        /*
         * +==============+
         * | CONSTRUCTORS |
         * +==============+
         */
        Line(InterestPoint point1, InterestPoint point2);

        Line(const Line &other);

        /*
         * +=========+
         * | DRAWING |
         * +=========+
         * Those are drawing primitives :
         * - _draw_ just draws the line on the canvas from start point to end point.
         * - _prolonger_ draws the line on the canvas until it meets the border of the canvas.
         */
        void draw(cv::Mat &canvas, const cv::Scalar& color = cv::Scalar(0, 0, 255), int thickness = 3) const;

        void prolonger(cv::Mat &canvas, cv::Scalar color = cv::Scalar(0, 255, 0), int thickness = 3) const;

        /*
         * +========+
         * | PRINTS |
         * +========+
         */
        void printPolar() const;
        void printPolar3D() const;
        void printCartes() const;

        std::string toString() const;

        std::string toStringCartes() const;

        std::string toStringDBSCAN() const;

        /*
         * +=============+
         * | MATHEMATICS |
         * +=============+
         */
        double distanceOrigine() const;

        Line getOrtho() const;

        cv::Vec2f director() const;

        cv::Point2f *intersect(const Line &other) const;

        void translate(double dx, double dy);

        double length() const;

        bool proche(const Line &other, int rhoTreshold = 8000, double thetaThreshold = 0.1) const;

        void invert();

        double pente() const;

        /*
         * +================+
         * | OVERRIDES / GS |
         * +================+
         */
        bool equals(const Line &other) const;

        const InterestPoint& getPoint1() const;

        const InterestPoint& getPoint2() const;

        double getTheta() const;

    private:
        cv::Point2f ortho() const;

        void computePolar();

        void computeCartes();

        InterestPoint _point1;
        InterestPoint _point2;

        /*
         * Représentation cartésienne ax + by + c = 0
         */
        double _a;
        double _b;
        double _c;

        cv::Point2f _origineOrtho;
        cv::Point2f _pointOrtho;

        /*
         * Représentation normale rho = xcos(theta) + ysin(theta)
         */
        double _rho;
        double _theta;
    };

    inline bool operator==(const Line &first, const Line &second) {
        return first.equals(second);
    }

    inline bool operator!=(const Line &first, const Line &second) {
        return !(first == second);
    }


    inline std::ostream &operator<<(std::ostream &os, const Line &line) {
        os << line.toString();
        return os;
    }


    cv::Mat fullPlane(const cv::Mat &src, std::vector<Line> &lines, int thickness = 3);
}
