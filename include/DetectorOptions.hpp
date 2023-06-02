//
// Created by rubytox on 27/07/2020.
//

#pragma once

#include <iostream>

struct DetectorOptions {
    std::string image;
    std::string rawName;
    std::string extension;
    std::string mask;

    int kp_hessian;

    double g2NN_angleThreshold;
    double g2NN_normThreshold;

    double length;

    int dbscan_minPts;
    double dbscan_epsilon;
    double dbscan_wx;
    double dbscan_wy;
    double dbscan_wtheta;

    double PSNR;

    bool draw_kp;
    bool draw_matches;
    bool draw_clusters;
    bool draw_hulls;

    bool stepByStep_expansion;
    bool before_dilation;

    int jobs;
};

