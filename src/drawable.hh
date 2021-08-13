#pragma once

#include "buffers.hh"

template<typename T = std::array<float, 3>>
struct drawable {
    vertex_array_object vao;
    vertex_buffer<T> vbo;
    index_buffer ibo;

    GLenum primitive;
    bool instanced;

    drawable(GLenum primitive_ = GL_POINTS, bool instanced_ = false):
        vbo({}, GL_DYNAMIC_DRAW),
        ibo({}, GL_DYNAMIC_DRAW),
        primitive(primitive_),
        instanced(instanced_)
    {
        if (!(primitive == GL_POINTS || primitive == GL_LINES || primitive == GL_TRIANGLES)) {
            throw std::runtime_error("error, primitive is not GL_POINTS, GL_LINES, or GL_TRIANGLES");
        }
    }

    template<typename F>
    void draw(F custom_params) {
        custom_params();
        vao.draw();
        vbo.draw();
        if (!instanced) {
            glDrawArrays(primitive, 0, vbo.data.size());
        } else {
            ibo.draw();
            glDrawElements(primitive, ibo.data.size(), GL_UNSIGNED_INT, 0);
        }
        glPointSize(1);
        glLineWidth(1);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    void draw() {
        draw([](){});
    }
};
