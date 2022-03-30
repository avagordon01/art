#include "engine/drawable.hh"
#include "drawables/grid-bezier-evaluator.hh"

struct grid_bezier {
    grid_bezier_renderer<2, glm::vec3> gbr;

    grid_bezier():
        gbr{GL_LINE, 32, 32}
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
        shared.inputs.view = glm::lookAt(
            glm::vec3(0, 0, 20),
            glm::vec3(0, 0, -10),
            glm::vec3(0, 1, 0)
        );
        shared.draw();

        gbr.draw();
    }
};
