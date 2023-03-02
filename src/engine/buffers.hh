#pragma once

#include <utility>
#include <vector>
#include "util/gl-data-layout.hh"

struct vertex_array_object {
    GLuint vao;
    vertex_array_object() {
        glCreateVertexArrays(1, &vao);
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
requires std::is_standard_layout_v<T>
struct buffer {
    using value_type = T;
    GLuint buffer_id;
    std::vector<T> data;
    T* previous_buffer;
    GLenum hint;

    buffer() = delete;
    buffer(std::vector<T> data_, GLenum hint_ = GL_STATIC_DRAW):
        data(data_),
        hint(hint_)
    {
        glCreateBuffers(1, &buffer_id);
        glBindBuffer(Target, buffer_id);
        glBufferData(Target, data.capacity() * sizeof(*data.data()), data.data(), hint);
        previous_buffer = data.data();
    }

    buffer(const buffer&) = delete;
    buffer &operator=(const buffer&) = delete;

    buffer(buffer &&other):
        buffer_id(std::exchange(other.buffer_id, 0)),
        data(std::move(other.data)),
        previous_buffer(std::exchange(other.previous_buffer, nullptr)),
        hint(std::exchange(other.hint, 0))
    {};
    buffer &operator=(buffer &&other) {
        if (this != &other) {
            glDeleteBuffers(1, &buffer_id);
            buffer_id = std::exchange(other.buffer_id, 0);
            data = std::move(other.data);
            previous_buffer = std::exchange(other.previous_buffer, nullptr);
            hint = std::exchange(other.hint, 0);
        }
        return *this;
    };

    template<typename D>
    void bind(GLuint program, const GLchar* name, D d) {
        if constexpr (Target == GL_ARRAY_BUFFER) {
            glBindBuffer(Target, buffer_id);
            GLint attrib_index = glGetAttribLocation(program, name);
            gl_vertex_attrib(attrib_index, d);
            glEnableVertexAttribArray(attrib_index);
        } else if constexpr (Target == GL_SHADER_STORAGE_BUFFER) {
            GLuint binding_index = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, name);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_index, buffer_id);
        } else if constexpr (Target == GL_UNIFORM_BUFFER) {
            GLuint binding_index = glGetUniformBlockIndex(program, name);
            glBindBufferBase(GL_UNIFORM_BUFFER, binding_index, buffer_id);
        }
    }

    void bind(GLuint program, const GLchar* name) {
        T d {};
        bind(program, name, d);
    }

    void draw() {
        glBindBuffer(Target, buffer_id);
        if (hint == GL_DYNAMIC_DRAW) {
            if (data.data() != previous_buffer) {
                glBufferData(Target, data.capacity() * sizeof(*data.data()), data.data(), hint);
                previous_buffer = data.data();
            } else {
                glBufferSubData(Target, 0, data.size() * sizeof(*data.data()), data.data());
            }
        } else if (hint == GL_DYNAMIC_READ) {
            //TODO handle the user changing the buffer size?
            glGetBufferSubData(Target, 0, data.size() * sizeof(*data.data()), data.data());
        } else if (hint == GL_DYNAMIC_COPY) {
            //XXX user can sort it out themselves
        } else {
            //don't need to do anything for GL_STREAM_* or GL_STATIC_*
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
template<typename T>
using storage_buffer = buffer<T, GL_SHADER_STORAGE_BUFFER>;
