//
// Created by rubytox on 24/06/2020.
//

#include "../include/ClusteredLine.hpp"

using namespace defals;

/**
 * Constructs a Line belonging to a cluster: it is a line that has an ID.
 *
 * @param line          The base Line object.
 * @param idCluster     The ID of the cluster the line belongs to.
 */
ClusteredLine::ClusteredLine(const defals::Line &line, int idCluster) : Line(line), _idCluster(idCluster) {
}

/**
 * Get the ID of the cluster the line belongs to.
 *
 * @return      The ID of the cluster the line belongs to.
 */
int ClusteredLine::getId() const {
    return _idCluster;
}

/**
 * Sets the ID of the cluster the line belongs to.
 *
 * @param id    The ID of the cluster the line belongs to.
 */
void ClusteredLine::setId(int id) {
    _idCluster = id;
}
