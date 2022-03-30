#include <vector>
#include <array>
#include <utility>
#include <random>
#include "engine/drawable.hh"
#include "engine/assimp.hh"
#include "drawables/ticks.hh"
#include "drawables/grid.hh"

struct brain {
    ticks ticks;
    grid grid;
    mesh mesh;
    drawable<> drawable;

    brain():
        ticks{},
        grid{},
        mesh{{
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/gray_left_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/gray_left_rsl_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/gray_right_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/gray_right_rsl_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/white_left_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/white_left_rsl_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/white_right_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/white_right_rsl_327680.obj",
        }, "brain"},
        drawable(R"foo(
in vec3 vertex;
out vec4 vertex_position;

out gl_PerVertex {
    vec4 gl_Position;
};
void main() {
    gl_Position = projection * view * vec4(vertex, 1.0f);
    vertex_position = projection * view * vec4(vertex, 1.0f);
}
)foo",
        R"foo(
in vec4 vertex_position;
in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    if (mod(time, 20) < 10) {
        colour = vec4(1) * int(gl_PrimitiveID < time * 6000);
    } else if (mod(time, 20) < 15) {
        colour = vec4(1) * float(int(vertex_position.y - time * 6) % 16 == 0);
    } else {
        colour = vec4(1);
    }
}
)foo")
    {
        drawable.vbo.data = std::move(mesh.vertices);
        drawable.ibo.data = std::move(mesh.indices);
        drawable.vbo.bind(drawable.shader.program_vertex, "vertex");
    }
    void draw() {
        float angle = 2 * M_PI * shared.inputs.time / 60;
        float distance = -100.0f;
        shared.inputs.view = glm::lookAt(
            glm::vec3(distance * sin(angle), 0.0f, distance * cos(angle)),
            glm::vec3(),
            glm::vec3(0, 1, 0)
        );

        shared.draw();
        grid.draw();
        ticks.draw();

        drawable.draw(GL_TRIANGLES, true, [](){
            glLineWidth(1);
            int t = static_cast<int>(shared.inputs.time) % 20;
            if (t >= 15 || (t >= 5 && t < 10)) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            } else {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            }
        });
    }
};
