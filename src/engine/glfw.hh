#pragma once
#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <string>

#include "engine/glerror.hh"

struct glfw_t {
    GLFWwindow* window;

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
    }

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

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetKeyCallback(window, key_callback);

        gl_setup_errors();
    }

    void draw(){
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
