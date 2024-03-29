#pragma once

#include <functional>
#include "buffers.hh"
#include "fullscreen-quad.hh"
#include "shader.hh"

template<typename T = std::array<float, 3>>
struct drawable {
    vertex_array_object vao;
    vertex_buffer<T> vbo;
    index_buffer ibo;
    shader shader;

    drawable(
        std::string vertex_source = shared_passthrough_vertex,
        std::string fragment_source = shared_passthrough_fragment
    ):
        vbo({}, GL_DYNAMIC_DRAW),
        ibo({}, GL_DYNAMIC_DRAW),
        shader(vertex_source, fragment_source)
    {
        vao.bind();
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

    void draw(GLenum primitive = GL_POINTS, bool instanced = false, std::function<void(void)> custom_params = [](){}) {
        vao.bind();
        shader.draw();
        if (!(primitive == GL_POINTS || primitive == GL_LINES || primitive == GL_TRIANGLES || primitive == GL_QUADS)) {
            throw std::runtime_error("error, primitive is not GL_POINTS, GL_LINES, GL_TRIANGLES, or GL_QUADS");
        }
        default_params();
        custom_params();
        if (primitive != GL_QUADS) {
            vbo.draw();
            if (!instanced) {
                glDrawArrays(primitive, 0, vbo.data.size());
            } else {
                ibo.draw();
                glDrawElements(primitive, ibo.data.size(), type_to_gltype<index_buffer::value_type>(), 0);
            }
        } else {
            fullscreen_quad{}.draw();
        }
        default_params();
    }
};
