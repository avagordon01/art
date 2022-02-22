#include "engine/drawable.hh"
#include "engine/rigid-body.hh"
#include "util/misc.hh"
#include "util/glm.hh"
#include <glm/ext.hpp>
#include <glm/gtx/string_cast.hpp>

struct icosphere {
    drawable<> drawable;
    rigid_body camera;

    icosphere():
        drawable{}
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
        drawable.vbo.bind(drawable.shader.program_vertex, "vertex");

        camera.angular_position = quaternionRand();
        camera.angular_velocity = quaternionRand() * 0.1f;
    }

    void draw() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera.tick();
        auto translate = glm::translate(glm::identity<glm::mat4>(), glm::vec3(0, 0, -4));
        shared.inputs.view = translate * glm::mat4_cast(camera.angular_position);

        shared.draw();

        drawable.draw(GL_TRIANGLES, true, [](){
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        });
    }
};
