//
// Created by rubytox on 24/06/2020.
//

#pragma once

#include "line.hpp"

namespace defals {

    /**
     * This class is used by DBSCAN scanner in order to represent a segment
     * belonging to a specific class.
     */
    class ClusteredLine : public Line {
    public:
        ClusteredLine(const Line& line, int idCluster = -1);

        int getId() const;
        void setId(int id);

    private:
        /**  -2 if noise, n > 0 otherwise  */
        int _idCluster;
    };
}
