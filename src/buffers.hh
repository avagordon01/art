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

template<typename T, GLenum Target>
struct buffer {
    GLuint buffer_id;
    std::vector<T> data;
    T* previous_buffer;
    bool immutable;
    buffer() {};
    buffer(std::vector<T> data_, bool immutable_ = true):
        data(data_),
        immutable(immutable_)
    {
        glGenBuffers(1, &buffer_id);
        glBindBuffer(Target, buffer_id);
        glBufferData(Target, data.capacity() * sizeof(*data.data()), data.data(), immutable ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        previous_buffer = data.data();
    }

    void bind(GLuint program, const GLchar* name) {
        if constexpr (Target == GL_ARRAY_BUFFER) {
            GLint attrib_index = glGetAttribLocation(program, name);
            glVertexAttribPointer(attrib_index, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
            glEnableVertexAttribArray(attrib_index);
        } else if constexpr (Target == GL_SHADER_STORAGE_BUFFER) {
            GLuint binding_index = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, name);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_index, buffer_id);
        } else if constexpr (Target == GL_UNIFORM_BUFFER) {
            GLuint binding_index = glGetUniformBlockIndex(program, name);
            glBindBufferBase(GL_UNIFORM_BUFFER, binding_index, buffer_id);
        }
    }

    void draw() {
        if (!immutable) {
            if (data.data() != previous_buffer) {
                glBufferData(Target, data.capacity() * sizeof(*data.data()), data.data(), immutable ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
            } else {
                glBufferSubData(Target, 0, data.size() * sizeof(*data.data()), data.data());
            }
        }
    }

    ~buffer() {
        glBindBuffer(Target, 0);
        glDeleteBuffers(1, &buffer_id);
    }
};

template<typename T>
using vertex_buffer = buffer<T, GL_ARRAY_BUFFER>;
using index_buffer = buffer<unsigned, GL_ELEMENT_ARRAY_BUFFER>;
template<typename T>
using uniform_buffer = buffer<T, GL_UNIFORM_BUFFER>;
