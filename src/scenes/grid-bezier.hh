#include "engine/drawable.hh"
#include "engine/shader.hh"
#include "drawables/grid-bezier-evaluator.hh"

struct grid_bezier {
    shared_uniforms& shared;
    grid_bezier_renderer<2, glm::vec3> gbr;

    grid_bezier(shared_uniforms& shared_):
        shared{shared_},
        gbr{shared, GL_LINE, 32, 32}
    {
        gbr.be.controls = {
            glm::vec3{0, 0, -10},
            glm::vec3{5, 5, -5},
            glm::vec3{10, 0, -10},

            glm::vec3{0, 5, -5},
            glm::vec3{5, 5, -5},
            glm::vec3{10, 5, -5},

            glm::vec3{0, 10, -10},
            glm::vec3{5, 5, -5},
            glm::vec3{10, 10, -10},
        };
    }
    void draw() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shared.inputs.view = glm::lookAt(
            glm::vec3(0, 0, 20),
            glm::vec3(0, 0, -10),
            glm::vec3(0, 1, 0)
        );
        shared.draw();

        gbr.draw();
    }
};
