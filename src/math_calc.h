#pragma once

#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include "tracerConfig.h"

#define PI		3.141592653589f
#define PI_MUL2 6.283185307179f
#define PI_DIV2 1.570796326794f
#define PI_MUL3 9.424777960769f
#define PI_DIV3 1.047197551196f
#define PI_MUL4 12.56637061435f
#define PI_DIV4 0.785398163397f
#define PI_INV  0.318309886184f

struct BoundingBox
{
    glm::vec3 min{  11111111,  11111111,  11111111 };
    glm::vec3 max{ -11111111, -11111111, -11111111 };

    void extend(glm::vec3 point) {
        if (point.x < this->min.x)
            this->min.x = point.x;
        if (point.y < this->min.y)
            this->min.y = point.y;
        if (point.z < this->min.z)
            this->min.z = point.z;

        if (point.x > this->max.x)
            this->max.x = point.x;
        if (point.y > this->max.y)
            this->max.y = point.y;
        if (point.z > this->max.z)
            this->max.z = point.z;
    }
};

inline glm::vec3 randomColor(int i)
{
    int r = unsigned(i) * 13 * 17 + 0x234235;
    int g = unsigned(i) * 7 * 3 * 5 + 0x773477;
    int b = unsigned(i) * 11 * 19 + 0x223766;
    return glm::vec3((r & 255) / 255.f,
        (g & 255) / 255.f,
        (b & 255) / 255.f);
}
