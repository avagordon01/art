#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdbool>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "font-atlas.hh"

static void error_callback(int error, const char* description) {
    std::cerr << "glfw error: " << description << std::endl;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    if (key == GLFW_KEY_F && action == GLFW_PRESS) {
        if (glfwGetWindowMonitor(window)) {
            glfwSetWindowMonitor(window, NULL, 0, 0, 800, 600, 60);
        } else {
            const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
            glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, mode->width, mode->height, 60);
        }
    }
}

std::string load(std::string filename) {
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

int main() {
    //setup glfw, window
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {
        std::cerr << "glfw error: failed to initialise" << std::endl;
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "project", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "glfw error: failed to create window" << std::endl;
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(window, GLFW_STICKY_KEYS, 1);
    glfwSetKeyCallback(window, key_callback);

    //create programs
    std::string error_log(4096, '\0');
    std::string s_header = load("src/header.glsl");
    std::string s_vertex = load("src/vertex.glsl");
    const char* shader_vertex[] = {
        "#version 450\n",
        s_header.c_str(),
        s_vertex.c_str(),
    };
    GLuint program_vertex = glCreateShaderProgramv(GL_VERTEX_SHADER, sizeof(shader_vertex) / sizeof(shader_vertex[0]), shader_vertex);
    glGetProgramInfoLog(program_vertex, 4096, NULL, error_log.data());
    if (error_log[0] != 0) {
        std::cerr << "vertex error:" << std::endl << error_log << std::endl;
        return 1;
    }
    std::string s_fragment = load("src/fragment.glsl");
    const char* shader_fragment[] = {
        "#version 450\n",
        s_header.c_str(),
        s_fragment.c_str(),
    };
    GLuint program_fragment = glCreateShaderProgramv(GL_FRAGMENT_SHADER, sizeof(shader_fragment) / sizeof(shader_fragment[0]), shader_fragment);
    glGetProgramInfoLog(program_fragment, 4096, NULL, error_log.data());
    if (error_log[0] != 0) {
        std::cerr << "fragment error:" << std::endl << error_log << std::endl;
        return 1;
    }
    GLuint pipeline_render;
    glGenProgramPipelines(1, &pipeline_render);
    glUseProgramStages(pipeline_render, GL_VERTEX_SHADER_BIT, program_vertex);
    glUseProgramStages(pipeline_render, GL_FRAGMENT_SHADER_BIT, program_fragment);
    glBindProgramPipeline(pipeline_render);

    monospace_printable_ascii_font_atlas atlas{
        "fonts/gohufont-2.1/gohufont-11.pcf.gz"
        //"fonts/artwiz-aleczapka-en-1.3/cure.pcf"
        //"fonts/tamsyn-font-1.11/Tamsyn6x12r.pcf"
    };
    glBindTextureUnit(0, atlas.texture());

    //create uniform buffer
    struct inputs {
        float mouse_x, mouse_y;
        float resolution_x, resolution_y;
        GLuint frame;
    } inputs;
    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(inputs), &inputs, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, glGetUniformBlockIndex(program_fragment, "inputs"), ubo);

    //create vertex array object
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(vao);

    glfwSetTime(0);
    for (uint64_t frame = 0;; frame++) {
        glfwPollEvents();
        if (glfwWindowShouldClose(window)) {
            std::cout << frame / glfwGetTime() << std::endl;
            glfwDestroyWindow(window);
            glfwTerminate();
            return 0;
        }

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        inputs.mouse_x += (mx - inputs.mouse_x) * 0.01;
        inputs.mouse_y += (my - inputs.mouse_y) * 0.01;
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        inputs.resolution_x = width;
        inputs.resolution_y = height;
        inputs.frame = frame;
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(inputs), &inputs);

        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();

        glfwSwapBuffers(window);
    }
}
