/**
 * @file    surf.hpp
 * This file defines some functions that will be used
 * to apply the SURF algorithm on images.
 */

#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "line.hpp"

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
[[deprecated]] cv::Mat surf_analysis(const cv::Mat& src, int minHessian = 400);

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
cv::Mat surf_matching(const cv::Mat& img1, const cv::Mat& img2,
                      int minHessian = 400);

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
cv::Mat surf_improved(const cv::Mat& img, int minHessian = 400);

/**
 * Helper function used in surf_matching algorithm: it checks
 * whether two points are near to each other.
 * In this function, p1 is near to p2 if p1 is one of the eight neighbours
 * of p2 or is equal to p2 :
 *
 * +---+----+---+
 * | a | b  | c |
 * +---+----+---+
 * | d | p2 | e |  p1 is near to p2 <==> p1 == p2 or p1 in [a, h]
 * +---+----+---+
 * | f | g  | h |
 * +---+----+---+
 *
 * Note: this relation is symmetric.
 *
 * @param   point1  The first point.
 * @param   point2  The second point.
 *
 * @return  True if point1 is not near to point2, false otherwise.
 */
bool areNotNear(const cv::Point2f& point1, const cv::Point2f& point2);

cv::Mat parallelLines(const cv::Mat& img);
