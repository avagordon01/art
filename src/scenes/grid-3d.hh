#include "engine/drawable.hh"
#include "engine/rigid-body.hh"

struct grid_3d {
    drawable<> drawable;
    rigid_body camera;

    grid_3d():
        drawable{}
    {
        int s = 5;
        std::vector<std::array<float, 3>> vertices {};
        for (float x = -s; x <= s; x++) {
            for (float y = -s; y <= s; y++) {
                for (float z = -s; z <= s; z++) {
                    vertices.push_back(std::array<float, 3>{x, y, z});
                    vertices.push_back(std::array<float, 3>{x, y, z+1});
                    vertices.push_back(std::array<float, 3>{x, y, z});
                    vertices.push_back(std::array<float, 3>{x, y+1, z});
                    vertices.push_back(std::array<float, 3>{x, y, z});
                    vertices.push_back(std::array<float, 3>{x+1, y, z});
                }
            }
        }
        //std::random_device r;
        //std::mt19937 g(r());
        //std::shuffle(vertices.begin(), vertices.end(), g);

        drawable.vbo.data = vertices;
        drawable.vbo.bind(drawable.shader.program_vertex, "vertex");

        camera.angular_position = quaternionRand();
        camera.angular_velocity = quaternionRand() * 0.1f;
    }
    void draw() {
        camera.angular_velocity *= 0.999;
        camera.velocity *= 0.999;
        camera.tick();
        auto translate = glm::translate(glm::identity<glm::mat4>(), glm::vec3(0, 0, -4));
        shared.inputs.view = translate * glm::mat4_cast(camera.angular_position);

        shared.draw();

        if (shared.inputs.time < 10 || (static_cast<int>(shared.inputs.time / 2) % 2) == 0) {
            drawable.draw(GL_LINES);
        } else {
            drawable.draw(GL_POINTS);
        }
    }
};
