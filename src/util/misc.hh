#pragma once

#include <array>

template<typename T>
T wrap(T x, T max, T min = 0) {
    return ((x - min) % (max - min)) + min;
}

int div_ceil(int numerator, int denominator) {
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
}

template<typename From>
std::array<float, 3> vec_to_array(From v) {
    return {v.x, v.y, v.z};
}

GLenum primitive_mode_to_primitive_type(GLenum mode) {
    if (mode == GL_FILL) {
        return GL_TRIANGLES;
    } else if (mode == GL_LINE) {
        return GL_LINES;
    } else if (mode == GL_POINT) {
        return GL_POINTS;
    } else {
        throw std::runtime_error("bad");
    }
}
