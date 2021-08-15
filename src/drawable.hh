#pragma once

#include "buffers.hh"

template<typename T = std::array<float, 3>>
struct drawable {
    vertex_array_object vao;
    vertex_buffer<T> vbo;
    index_buffer ibo;
    fullscreen_quad quad;

    GLenum primitive;
    bool instanced;

    drawable(GLenum primitive_ = GL_POINTS, bool instanced_ = false):
        vbo({}, GL_DYNAMIC_DRAW),
        ibo({}, GL_DYNAMIC_DRAW),
        primitive(primitive_),
        instanced(instanced_)
    {
        if (!(primitive == GL_POINTS || primitive == GL_LINES || primitive == GL_TRIANGLES || primitive == GL_QUADS)) {
            throw std::runtime_error("error, primitive is not GL_POINTS, GL_LINES, GL_TRIANGLES, or GL_QUADS");
        }
    }
    void default_params() {
        glPointSize(1);
        glLineWidth(1);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);
        glDisable(GL_SCISSOR_TEST);
        glEnable(GL_FRAMEBUFFER_SRGB);
    }

    template<typename F>
    void draw(F custom_params) {
        default_params();
        custom_params();
        vao.draw();
        if (primitive != GL_QUADS) {
            vbo.draw();
            if (!instanced) {
                glDrawArrays(primitive, 0, vbo.data.size());
            } else {
                ibo.draw();
                glDrawElements(primitive, ibo.data.size(), GL_UNSIGNED_INT, 0);
            }
        } else {
            quad.draw();
        }
        default_params();
    }
    void draw() {
        draw([](){});
    }
};
