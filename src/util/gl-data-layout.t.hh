#pragma once

#include <array>
#include <vector>

void test() {
    {
        struct char_vertex {
            std::array<float, 2> vertex;
            std::array<float, 3> texcoords;
            uint32_t other;
        };
        gl_vertex_attrib(0, &char_vertex::vertex);
        gl_vertex_attrib(0, &char_vertex::texcoords);
        gl_vertex_attrib(0, &char_vertex::other);
    }

    {
        std::vector<std::array<float, 3>> data;
        gl_vertex_attrib(0, data.front());
    }

    {
        std::vector<uint32_t> data;
        gl_vertex_attrib(0, data.front());
    }
}
