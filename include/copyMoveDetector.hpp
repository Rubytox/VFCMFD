#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <thread>
#include <fstream>
#include <regex>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <boost/log/trivial.hpp>

#include "line.hpp"
#include "InterestPoint.hpp"
#include "InterestPoints.hpp"
#include "dbscan.hpp"
#include "ClusteredLine.hpp"
#include "DetectorOptions.hpp"

struct DetectorOptions;

namespace defals {

/**
 * This will help understand the code better.
 */
    using Cluster = std::vector<Line>;

/**
 * This class represents a copy/move forgery detector.
 * It provided the user with one main function _detect_ which
 * processes the image, and a function _show_ which displays
 * and saves the processed image.
 */
    class copyMoveDetector {
    public:
        copyMoveDetector(const DetectorOptions& options);

        void detect();

        void printLines() const;

        void printClusters() const;

        void printInfo();

        void show() const;

        void randomLines();

    private:
        void computeKeypoints();
        void computeMatches();
        void computeMatch(int i);
        friend void runMatches(copyMoveDetector &detector, int start, int end);

        void computeBetterMatches();
        void computeBetterMatch(int i);
        friend void runBetterMatches(copyMoveDetector &detector, int start, int end);

        void computeLines();
        void computeClusters();
        void computeHull();
        void computeMask(int kernelSize = 3);

        double computePSNR(int i) const;
        std::vector<std::pair<cv::Point, cv::Point>> borderOfHull(int i) const;
        std::vector<std::pair<cv::Point, cv::Point>> borderOfHull(const std::vector<std::pair<cv::Point, cv::Point>>& hull) const;
        void extendMask();
        std::vector<std::pair<cv::Point, cv::Point>> checkEQM(const cv::Point &pt1,
                                                              const cv::Point &pt2,
                                                              bool& border,
                                                              const cv::Size &ksize = cv::Size(7, 7),
                                                              const double PSNR_threshold = 100);

        void conclude() const;

        double computeDice() const;
        void computeFScore(double& precision, double& recall, double& F1) const;

        inline static void save(const cv::Mat &img, const std::string &filename = "lines");

        DetectorOptions _options;

        cv::Mat _image;
        cv::Mat _mask;

        InterestPoints _interestPoints;

        std::vector<std::vector<InterestPoint>> _allMatches;
        std::vector<Line> _lines;

        std::vector<Cluster> _clusters;
        std::vector<Line> _outliers;

        std::vector<std::vector<std::pair<InterestPoint, Line>>> _hulls;
        cv::Mat _computedMask;
        cv::Mat _extendedMask;
    };

    void runMatches(copyMoveDetector &detector, int start, int end);
    void runBetterMatches(copyMoveDetector& detector, int start, int end);
}