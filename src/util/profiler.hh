#pragma once
#include <chrono>

struct profiler {
    using clock_type = std::chrono::high_resolution_clock;
    std::vector<std::chrono::time_point<clock_type>> times;
    std::vector<std::string> names;

    profiler() {
        capture("start");
    }

    void capture(std::string name) {
        times.push_back(clock_type::now());
        names.push_back(name);
    }

    ~profiler() {
        for (size_t i = 0, j = 1; i < times.size() && j < times.size(); i++, j++) {
            using millis = std::chrono::duration<float, std::milli>;
            std::cout << names[j] << ": " << std::chrono::duration_cast<millis>(times[j] - times[i]).count() << "ms" << std::endl;
        }
    }
};
