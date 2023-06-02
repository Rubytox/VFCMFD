#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>

#include "line.hpp"
#include "ClusteredLine.hpp"

#define UNCLASSIFIED -1
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

/**
 * This class stands for a scanner using DBSCAN algorithm.
 * It specializes in the detection of dense clusters of segments
 * represented in 4D by:
 *          (x, y, theta, l)
 * where:
 * - (x,y) : closest end of the segment to (0, 0)
 * - theta : the polar angle of the segment
 * - l : the length of the segment
 */
class DBSCAN {
public:
    /*
     * +==============+
     * | CONSTRUCTORS |
     * +==============+
     */
    DBSCAN(unsigned int minPts, double eps, std::vector<defals::ClusteredLine>& lines,
           int height, int width,
           double wx, double wy, double wtheta);

    /*
     * +=============+
     * |  ALGORITHM  |
     * +=============+
     */
    std::vector<defals::ClusteredLine> run();

    /**
     * This function computes the distance between two segments defined as said above. Each of the four parameters
     * has a weight that can be used to give more or less importance to one parameter.
     *
     * @param pointCore     One of the lines.
     * @param pointTarget   The other line.
     * @param wx            The weight on parameter x.
     * @param wy            The weight on parameter y.
     * @param wtheta        The weight on parameter theta.
     * @param wl            The weight on parameter l.
     *
     * @return              The weighted distance of the two lines.
     */
    inline double calculateDistance(const defals::ClusteredLine& pointCore, const defals::ClusteredLine& pointTarget) {
        int sizeRatio = (_height + _width) / 2;
        double wl = 1 - _wx - _wy - _wtheta;

        return sqrt(_wx * pow(pointCore.getPoint1().pt.x - pointTarget.getPoint1().pt.x, 2) / sizeRatio +
                    _wy * pow(pointCore.getPoint1().pt.y - pointTarget.getPoint1().pt.y, 2) / sizeRatio +
                    _wtheta * pow(pointCore.getTheta() - pointTarget.getTheta(), 2) / pointCore.getTheta() +
                       wl * pow(pointCore.length() - pointTarget.length(), 2) / pointCore.length());
    }

private:
    std::vector<int> calculateCluster(defals::ClusteredLine& point);
    int expandCluster(defals::ClusteredLine& line, int clusterID);

    /**  The lines we want to cluster  */
    std::vector<defals::ClusteredLine> _lines;
    /**  The minimal number of points in a neighbourhood */
    unsigned int _minPoints;
    /**  The radius of the considered neighbourhood */
    double _epsilon;

    int _height;
    int _width;

    double _wx, _wy, _wtheta;
};

#endif // DBSCAN_H
