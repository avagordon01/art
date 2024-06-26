#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <assert.h>

#include <iostream>
#include <stdexcept>

static void error_callback(int error, const char *description) {
    throw std::runtime_error("glfw error: " + std::string(description));
}

int main() {
    glfwSetErrorCallback(error_callback);
    for (auto platform: {GLFW_PLATFORM_X11, GLFW_PLATFORM_WAYLAND}) {
        assert(glfwPlatformSupported(platform));
        glfwInitHint(GLFW_PLATFORM, platform);
        if (!glfwInit()) {
            throw std::runtime_error("glfw error: failed to initialise");
        }
        assert(glfwGetPlatform() == platform);
        GLFWwindow *window = glfwCreateWindow(640, 480, "project", NULL, NULL);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("glfw error: failed to create window");
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        glfwShowWindow(window);
        glfwFocusWindow(window);

        for (size_t i = 0; i < 64; i++) {
            std::cerr << "frame " << i << std::endl;
            glClearColor(1.0, 0.0, 0.0, 1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glfwPollEvents();
            std::cerr << "before glfwSwapBuffers" << std::endl;
            glfwSwapBuffers(window);
            std::cerr << "after glfwSwapBuffers" << std::endl;
            glfwPollEvents();
        }

        if (window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }
}
