#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <string>

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

    void draw(){
        glfwSwapBuffers(window);
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glfwPollEvents();
    }

    ~glfw_t() {
        if (window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }
};
