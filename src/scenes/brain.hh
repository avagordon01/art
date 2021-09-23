#include <vector>
#include <array>
#include <utility>
#include <random>
#include "engine/buffers.hh"
#include "engine/shader.hh"
#include "engine/shared-uniforms.hh"
#include "engine/shader.hh"
#include "engine/drawable.hh"

struct brain {
    glfw_t& glfw;
    shared_uniforms shared;

    ticks ticks;
    grid grid;
    mesh mesh;
    drawable<> drawable;
    shader shader;

    brain(glfw_t& glfw_):
        glfw{glfw_},
        shared{glfw},
        ticks{shared},
        grid{shared},
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
        drawable(GL_TRIANGLES, true),
        shader(shared.header_shader_text + R"foo(
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
        shared.header_shader_text + R"foo(
in vec4 vertex_position;
in vec4 gl_FragCoord;
out vec4 colour;

void main() {)foo" +
    ((scene == scenes::dot_brain) ? R"foo(
    colour = vec4(1);
}
    )foo" : R"foo(
    if (mod(time, 10) <= 5) {
        colour = vec4(1) * int(gl_PrimitiveID < time * 6000);
    } else {
        colour = vec4(1) * float(int(vertex_position.y - time * 6) % 16 == 0);
    }
}
)foo"))
    {
        drawable.vbo.data = mesh.vertices;
        drawable.ibo.data = mesh.indices;
        drawable.vbo.bind(shader.program_vertex, "vertex");
        shared.bind(shader.program_vertex);
        shared.bind(shader.program_fragment);
    }
    void draw() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int w, h;
        glfwGetWindowSize(glfw.window, &w, &h);
        shared.inputs.projection = glm::perspective(glm::radians(75.0f), static_cast<float>(w) / h, 0.1f, 200.f);
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

        shader.draw();
        drawable.draw([](){
            glLineWidth(1);
            if (scene == scenes::dot_brain) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            } else {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            }
        });
    }
};
