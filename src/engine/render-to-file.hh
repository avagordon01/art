#include <cstdio>

struct render_to_file {
    std::vector<uint8_t> buffer;
    FILE* ffmpeg = NULL;
    int w, h;

    render_to_file(glfw_t &glfw, std::string filename) {
        glfwGetWindowSize(glfw.window, &w, &h);

        using namespace std::string_literals;
        std::string command = 
            "ffmpeg -y -f rawvideo -s "s +
            std::to_string(w) + "x"s + std::to_string(h) +
            " -pix_fmt rgb24 -r 60 -i - -vf vflip -an -b:v 1000k "s +
            filename;
        ffmpeg = popen(command.c_str(), "w");
        if (!ffmpeg) {
            throw std::runtime_error("bad");
        }

        buffer.resize(w * h * 3);
    }
    ~render_to_file() {
        pclose(ffmpeg);
    }

    void draw() {
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
        fwrite(buffer.data(), buffer.size(), 1, ffmpeg);
    }
};
