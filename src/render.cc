#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdbool>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "font-atlas.hh"

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
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

GLuint create_program(GLenum type, std::string program_text) {
    const char* s[] = {
        program_text.c_str(),
    };
    GLuint program = glCreateShaderProgramv(type, 1, s);
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
        //setup glfw, window
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
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    ~glfw_t() {
        if (window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }
};

int main() {
    glfw_t glfw;

    glfwSwapInterval(1);
    glfwSetInputMode(glfw.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(glfw.window, GLFW_STICKY_KEYS, 1);
    glfwSetKeyCallback(glfw.window, key_callback);

    //create programs
    GLuint program_vertex = create_program(GL_VERTEX_SHADER, load("src/vertex.glsl"));
    GLuint program_fragment = create_program(GL_FRAGMENT_SHADER, load("src/fragment.glsl"));

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
    uint64_t frame = 0;
    for (; !glfwWindowShouldClose(glfw.window); frame++) {
        double mx, my;
        glfwGetCursorPos(glfw.window, &mx, &my);
        inputs.mouse_x += (mx - inputs.mouse_x) * 0.01;
        inputs.mouse_y += (my - inputs.mouse_y) * 0.01;
        int width, height;
        glfwGetWindowSize(glfw.window, &width, &height);
        glViewport(0, 0, width, height);
        inputs.resolution_x = width;
        inputs.resolution_y = height;
        inputs.frame = frame;
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(inputs), &inputs);

        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();

        glfw.tick();
    }
    std::cout << frame / glfwGetTime() << std::endl;
}
