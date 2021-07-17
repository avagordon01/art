#pragma once

struct vertex_array_object {
    GLuint vao;
    vertex_array_object() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }
    void draw() {
        glBindVertexArray(vao);
    }
    ~vertex_array_object() {
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &vao);
    }
};

template<typename T>
struct vertex_buffer {
    GLuint vbo;
    std::vector<T> data;
    size_t buffer_size;
    bool immutable;
    vertex_buffer() {};
    vertex_buffer(std::vector<T> data_, GLuint vertex_attrib_index, bool immutable_ = true):
        data(data_),
        immutable(immutable_)
    {
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(*data.data()), data.data(), immutable ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(vertex_attrib_index, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(vertex_attrib_index);
        buffer_size = data.size();
    }

    void draw() {
        if (!immutable) {
            if (buffer_size != data.size()) {
                //TODO resize(/recreate) buffer
            }
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.size() * sizeof(*data.data()), data.data());
        }
    }

    ~vertex_buffer() {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &vbo);
    }
};

struct index_buffer {
    GLuint ibo;
    std::vector<unsigned> data;
    size_t buffer_size;
    bool immutable;
    index_buffer() {};
    index_buffer(std::vector<unsigned> data_, bool immutable_ = true):
        data(data_),
        immutable(immutable_)
    {
        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(*data.data()), data.data(), immutable ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        buffer_size = data.size();
    }

    void draw() {
        if (!immutable) {
            if (buffer_size != data.size()) {
                //TODO resize(/recreate) buffer
            }
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.size() * sizeof(*data.data()), data.data());
        }
    }

    ~index_buffer() {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &ibo);
    }
};
