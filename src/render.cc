#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdbool>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include "font-atlas.hh"
#include "layer.hh"

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

std::string read_file_as_string(std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string buffer(size, '\0');
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("error reading file");
    }
    buffer += '\0';
    return buffer;
}

template<typename ...S>
GLuint create_program(GLenum type, S... program_texts) {
    const char* s[] = {
        "#version 450\n",
        program_texts.c_str()...
    };
    GLuint program = glCreateShaderProgramv(type, 1 + sizeof...(program_texts), s);
    std::string error_log(4096, '\0');
    glGetProgramInfoLog(program, 4096, NULL, error_log.data());
    if (error_log[0] != 0) {
        std::cerr << "vertex error:" << std::endl << error_log << std::endl;
        return 1;
    }
    return program;
}

struct glfw_t {
    GLFWwindow* window;

    static void error_callback(int error, const char* description) {
        throw std::runtime_error("glfw error: " + std::string(description));
    }

    glfw_t() {
        glfwSetErrorCallback(error_callback);
        if (!glfwInit()) {
            throw std::runtime_error("glfw error: failed to initialise");
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        window = glfwCreateWindow(mode->width, mode->height, "project", monitor, NULL);

        if (!window) {
            glfwTerminate();
            throw std::runtime_error("glfw error: failed to create window");
        }
        glfwMakeContextCurrent(window);
    }

    void tick(){
        glFinish();
        glfwSwapBuffers(window);
        glfwPollEvents();
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
    }

    ~glfw_t() {
        if (window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }
};

struct shared_uniforms : layer_t {
    struct inputs {
        float mouse_x, mouse_y;
        float resolution_x, resolution_y;
        GLuint frame;
    } inputs;
    std::string header_shader_text = R"foo(
layout(std140, binding=0) uniform inputs {
    vec2 mouse;
    vec2 resolution;
    uint frame;
};
)foo";
    glfw_t& glfw;
    shared_uniforms(glfw_t &glfw_):
        glfw(glfw_)
    {
        inputs.frame = 0;

        GLuint ubo;
        glGenBuffers(1, &ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(inputs), &inputs, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    }
    void draw() override {
        double mx, my;
        glfwGetCursorPos(glfw.window, &mx, &my);
        inputs.mouse_x = mx;
        inputs.mouse_y = my;
        int w, h;
        glfwGetWindowSize(glfw.window, &w, &h);
        inputs.resolution_x = w;
        inputs.resolution_y = h;
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(inputs), &inputs);

        inputs.frame++;
    }
};

struct juliaset : layer_t {
    GLuint program_vertex, program_fragment, pipeline_render, vao;
    template<typename ...S>
    juliaset(S... program_texts) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(vao);

        program_vertex = create_program(GL_VERTEX_SHADER, read_file_as_string("src/julia.vert.glsl"));
        program_fragment = create_program(GL_FRAGMENT_SHADER, program_texts..., read_file_as_string("src/julia.frag.glsl"));

        glGenProgramPipelines(1, &pipeline_render);
        glUseProgramStages(pipeline_render, GL_VERTEX_SHADER_BIT, program_vertex);
        glUseProgramStages(pipeline_render, GL_FRAGMENT_SHADER_BIT, program_fragment);

        glBindVertexArray(0);
    }
    void draw() override {
        glBindVertexArray(vao);
        glBindProgramPipeline(pipeline_render);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
};

struct text_overlay : layer_t {
    GLuint program_vertex, program_fragment, pipeline_render, vao;
    template<typename ...S>
    text_overlay(S... program_texts) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(vao);

        program_vertex = create_program(GL_VERTEX_SHADER, read_file_as_string("src/text.vert.glsl"));
        program_fragment = create_program(GL_FRAGMENT_SHADER, program_texts..., read_file_as_string("src/text.frag.glsl"));

        glGenProgramPipelines(1, &pipeline_render);
        glUseProgramStages(pipeline_render, GL_VERTEX_SHADER_BIT, program_vertex);
        glUseProgramStages(pipeline_render, GL_FRAGMENT_SHADER_BIT, program_fragment);
        glBindProgramPipeline(pipeline_render);

        glBindVertexArray(0);
    }
    void draw() override {
        glBindVertexArray(vao);
        glBindProgramPipeline(pipeline_render);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
};

int main() {
    glfw_t glfw;

    glfwSwapInterval(1);
    glfwSetInputMode(glfw.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(glfw.window, GLFW_STICKY_KEYS, 1);
    glfwSetKeyCallback(glfw.window, key_callback);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    shared_uniforms s{glfw};
    monospace_printable_ascii_font_atlas atlas{
        "fonts/gohufont-2.1/gohufont-11.pcf.gz"
        //"fonts/artwiz-aleczapka-en-1.3/cure.pcf"
        //"fonts/tamsyn-font-1.11/Tamsyn6x12r.pcf"
    };
    juliaset j{s.header_shader_text, atlas.header_shader_text};
    text_overlay t{s.header_shader_text, atlas.header_shader_text};

    glfwSetTime(0);
    while (!glfwWindowShouldClose(glfw.window)) {
        s.draw();
        j.draw();
        t.draw();

        glfw.tick();
    }
    std::cout << s.inputs.frame / glfwGetTime() << std::endl;
}
