#include "engine/shader.hh"
#include "engine/drawable.hh"

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

template<size_t Order>
std::array<float, Order + 1> bernstein_coefficients(float x) {
    std::array<float, Order + 1> b;
    if (Order == 0) {
        b[0] = 1.0;
    } else {
        b[0] = 1.0 - x;
        b[1] = x;

        for (size_t i = 2; i <= Order; i++) {
            b[i] = x * b[i-1];
            for (size_t j = i - 1; 1 <= j; j--) {
                b[j] = x * b[j-1] + (1.0 - x) * b[j];
            }
            b[0] = (1.0 - x) * b[0];
        }
    }
    return b;
}

template<typename From>
std::array<float, 3> vec_to_array(From x) {
    return {x[0], x[1], x[2]};
}

template<size_t Order, typename T>
struct grid_bezier_evaluator {
    GLenum type;
    drawable<> drawable;
    shader shader;
    size_t n, m;
    std::array<T, (Order + 1) * (Order + 1)> controls;
    grid_bezier_evaluator(shared_uniforms& shared_uniforms, GLenum mode, size_t n_, size_t m_):
        type(primitive_mode_to_primitive_type(mode)),
        drawable(),
        shader(shared_uniforms.header_shader_text + shared_uniforms.passthrough_vertex, shared_uniforms.passthrough_fragment),
        n(n_), m(m_)
    {
        shared_uniforms.bind(shader.program_vertex);
        drawable.vbo.bind(shader.program_vertex, "vertex");
    }
    std::array<float, 3> eval_coord(size_t i, size_t j) {
        float u = n == 0 ? 0 : static_cast<float>(i) / n;
        float v = m == 0 ? 0 : static_cast<float>(j) / m;
        auto us = bernstein_coefficients<Order>(u);
        auto vs = bernstein_coefficients<Order>(v);
        T result{0};
        for (size_t i = 0; i < Order + 1; i++) {
            for (size_t j = 0; j < Order + 1; j++) {
                result += us[i] * vs[j] * controls[i + j * (Order + 1)];
            }
        }
        return vec_to_array(result);
    }
    void draw() {
        drawable.vbo.data.clear();
        if (type == GL_TRIANGLES) {
            for (size_t j = 0; j < m; j++) {
                for (size_t i = 0; i <= n; i++) {
                    drawable.vbo.data.push_back(eval_coord(i, j));
                    drawable.vbo.data.push_back(eval_coord(i, j + 1));
                    drawable.vbo.data.push_back(eval_coord(i + 1, j));

                    drawable.vbo.data.push_back(eval_coord(i + 1, j + 1));
                    drawable.vbo.data.push_back(eval_coord(i + 1, j));
                    drawable.vbo.data.push_back(eval_coord(i, j + 1));
                }
            }
        } else if (type == GL_LINES) {
            for (size_t j = 0; j <= m; j++) {
                for (size_t i = 0; i < n; i++) {
                    drawable.vbo.data.push_back(eval_coord(i, j));
                    drawable.vbo.data.push_back(eval_coord(i + 1, j));
                }
            }

            for (size_t i = 0;  i <= n; i++) {
                for (size_t j = 0; j < m; j++) {
                    drawable.vbo.data.push_back(eval_coord(i, j));
                    drawable.vbo.data.push_back(eval_coord(i, j + 1));
                }
            }
        } else if (type == GL_POINTS) {
            for (size_t j = 0; j <= m; j++) {
                for (size_t i = 0; i <= n; i++) {
                    drawable.vbo.data.push_back(eval_coord(i, j));
                }
            }
        }
        shader.draw();
        drawable.draw(type);
    }
};
