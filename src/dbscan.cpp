#include "../include/dbscan.hpp"

using namespace std;
using namespace defals;

/**
 * Constructs a scanner using DBSCAN algorithm.
 *
 * @param minPts    Minimal number of points required in a eps-neighbourhood.
 * @param eps       Radius of the neighbourhood.
 * @param lines     The lines to cluster.
 */
DBSCAN::DBSCAN(unsigned int minPts, double eps, std::vector<defals::ClusteredLine>& lines,
               int height, int width,
               double wx, double wy, double wtheta) {
    _minPoints = minPts;
    _epsilon = eps;
    _lines = lines;
    _height = height;
    _width = width;
    _wx = wx;
    _wy = wy;
    _wtheta = wtheta;
}

/**
 * This is the method that starts the DBSCAN algorithm.
 *
 * @return  A vector containing the lines with their _idCluster set.
 */
vector<ClusteredLine> DBSCAN::run()
{
    int clusterID = 1;
    vector<ClusteredLine>::iterator iter;
    for(iter = _lines.begin(); iter != _lines.end(); ++iter)
    {
        if (iter->getId() == UNCLASSIFIED)
        {
            if (expandCluster(*iter, clusterID) != FAILURE)
                clusterID++;
        }
    }
    return _lines;
}

/**
 * Given a line, creates new cluster from it, adds line to another cluster
 * or classifies it as noise.
 *
 * @param line          A line that hasn't been clustered yet.
 * @param clusterID     The ID of the cluster we're expanding.
 *
 * @return      SUCCESS if the line has been successfully classified.
 *              FAILRUE if the line has been classified as noise.
 */
int DBSCAN::expandCluster(ClusteredLine& line, int clusterID) {
    /*
     * Compute the eps-neighbourhood of _line_.
     */
    vector<int> clusterSeeds = calculateCluster(line);

    /*
     * If there aren't enough lines in the eps-neighbourhood,
     * we classify it as noise.
     */
    if (clusterSeeds.size() < _minPoints)
    {
        line.setId(NOISE);
        return FAILURE;
    }

    int index = 0, indexCorePoint = 0;
    vector<int>::iterator iterSeeds;
    for (iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
    {
        _lines.at(*iterSeeds).setId(clusterID);
        if (_lines.at(*iterSeeds).getPoint1().pt.x == line.getPoint1().pt.x &&
            _lines.at(*iterSeeds).getPoint1().pt.y == line.getPoint1().pt.y &&
            _lines.at(*iterSeeds).getTheta() == line.getTheta() &&
            _lines.at(*iterSeeds).length() == line.length())
        {
            indexCorePoint = index;
        }
        ++index;
    }
    clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

    for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
    {
        vector<int> clusterNeighors = calculateCluster(_lines.at(clusterSeeds[i]));

        if ( clusterNeighors.size() >= _minPoints )
        {
            vector<int>::iterator iterNeighors;
            for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
            {
                if ( _lines.at(*iterNeighors).getId() == UNCLASSIFIED || _lines.at(*iterNeighors).getId() == NOISE )
                {
                    if ( _lines.at(*iterNeighors).getId() == UNCLASSIFIED )
                    {
                        clusterSeeds.push_back(*iterNeighors);
                        n = clusterSeeds.size();
                    }
                    _lines.at(*iterNeighors).setId(clusterID);
                }
            }
        }
    }

    return SUCCESS;
}

/**
 * Computes the eps-neighbourhood of a line.
 *
 * @param point     The line at the center of the neighbourhood.
 *
 * @return      A vector of indices representing the neighbour of _line_ in __lines_.
 */
vector<int> DBSCAN::calculateCluster(ClusteredLine& point)
{
    int index = 0;
    vector<ClusteredLine>::iterator iter;
    vector<int> clusterIndex;
    for (iter = _lines.begin(); iter != _lines.end(); ++iter) {
        double distance = calculateDistance(point, *iter);

        if (distance <= _epsilon)
            clusterIndex.push_back(index);

        index++;
    }
    return clusterIndex;
}


