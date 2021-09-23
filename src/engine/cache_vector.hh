#pragma once
#include <fstream>
#include <vector>
#include <type_traits>
#include <optional>
#include <filesystem>

template<typename T>
std::optional<std::vector<T>> vector_from_file(std::string filename) {
    std::ifstream file(filename, std::ios::in | std::ifstream::binary);
    if (file.fail()) {
        return std::nullopt;
    }
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::vector<T> vector;
    vector.resize(size);
    file.read(reinterpret_cast<char*>(vector.data()), size * sizeof(T));
    return vector;
}

template<typename T>
void vector_to_file(std::string filename, std::vector<T>& vector) {
    std::filesystem::path path(filename);
    path.remove_filename();
    std::filesystem::create_directory(path);
    static_assert(std::is_trivially_copyable<T>::value, "vector<T> T must be trivially_copyable");
    std::ofstream file(filename, std::ios::out | std::ofstream::binary);
    size_t size = vector.size();
    file.write(reinterpret_cast<char*>(&size), sizeof(size));
    file.write(reinterpret_cast<char*>(vector.data()), size * sizeof(T));
}
