#include "engine/drawable.hh"
#include "engine/shader.hh"
#include "engine/shared-uniforms.hh"
#include "util/misc.hh"

struct icosphere {
    glfw_t& glfw;
    shared_uniforms shared;
    drawable<> drawable;
    shader shader;

    icosphere(glfw_t& glfw_):
        glfw{glfw_},
        shared{glfw},
        drawable{},
        shader{shared.header_shader_text + shared.passthrough_vertex, shared.passthrough_fragment}
    {
        using std::numbers::pi;
        const float h_angle = pi / 180 * 72;
        const float v_angle = atanf(1.0f / 2);
        const float radius = 1.0f;

        std::vector<std::array<float, 3>> vertices(12);
        std::vector<unsigned> indices;
        vertices[0][0] = 0;
        vertices[0][1] = 0;
        vertices[0][2] = radius;

        for (size_t i = 1; i <= 5; i++) {
            size_t j = i + 5;

            float z = radius * sinf(v_angle);
            float xy = radius * cosf(v_angle);

            vertices[i][0] = xy * cosf(-pi / 2 + h_angle * (i - 1 - 0.5));
            vertices[j][0] = xy * cosf(-pi / 2 + h_angle * (i - 1));
            vertices[i][1] = xy * sinf(-pi / 2 + h_angle * (i - 1 - 0.5));
            vertices[j][1] = xy * sinf(-pi / 2 + h_angle * (i - 1));
            vertices[i][2] = z;
            vertices[j][2] = -z;
        }

        vertices[11][0] = 0;
        vertices[11][1] = 0;
        vertices[11][2] = -radius;

        {
            for (size_t t = 0; t < 5; t++) {
                indices.push_back(0);
                indices.push_back(t+1);
                indices.push_back(wrap<size_t>(t+2, 6, 1));
            }
            for (size_t t = 0; t < 5; t++) {
                indices.push_back(t+1);
                indices.push_back(t+6);
                indices.push_back(wrap<size_t>(t+2, 6, 1));
            }
            for (size_t t = 0; t < 5; t++) {
                indices.push_back(t+6);
                indices.push_back(wrap<size_t>(t+2, 6, 1));
                indices.push_back(wrap<size_t>(t+7, 11, 6));
            }
            for (size_t t = 0; t < 5; t++) {
                indices.push_back(11);
                indices.push_back(t+6);
                indices.push_back(wrap<size_t>(t+7, 11, 6));
            }
        }

        drawable.vbo.data = vertices;
        drawable.ibo.data = indices;
        drawable.vbo.bind(shader.program_vertex, "vertex");
        shared.bind(shader.program_vertex);
    }
    void draw() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int w, h;
        glfwGetWindowSize(glfw.window, &w, &h);
        shared.inputs.projection = glm::perspective(glm::radians(75.0f), static_cast<float>(w) / h, 0.1f, 200.f);
        float angle = 2 * M_PI * shared.inputs.time / 60;
        float distance = -4.0f;
        shared.inputs.view = glm::lookAt(
            glm::vec3(distance * sin(angle), distance * cos(angle), 0.0f),
            glm::vec3(),
            glm::vec3(0, 0, 1)
        );

        shared.draw();

        shader.draw();
        drawable.draw(GL_TRIANGLES, true, [](){
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        });
    }
};
