#ifndef DATAPOINT_H
#define DATAPOINT_H

#include <array>
#include <cstdint>

#include "config.h"

struct DataPoint {
    IMAGE_SHAPE pixels;
    uint8_t label;
};

#endif