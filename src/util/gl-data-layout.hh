#pragma once

template <typename T>
GLenum type_to_gltype() {
	if (typeid(T) == typeid(float)) {
		return GL_FLOAT;
    } else if (typeid(T) == typeid(double)) {
		return GL_DOUBLE;
    } else if (typeid(T) == typeid(char)) {
        return GL_BYTE;
    } else if (typeid(T) == typeid(unsigned char)) {
		return GL_UNSIGNED_BYTE;
    } else if (typeid(T) == typeid(int)) {
		return GL_INT;
    } else if (typeid(T) == typeid(unsigned int)) {
		return GL_UNSIGNED_INT;
    } else if (typeid(T) == typeid(short)) {
		return GL_SHORT;
    } else if (typeid(T) == typeid(unsigned short)) {
		return GL_UNSIGNED_SHORT;
	} else {
        return GL_INVALID_ENUM;
    }
}

//https://gist.github.com/graphitemaster/494f21190bb2c63c5516
template<typename T, typename M>
inline size_t constexpr offset_of(M T::*member) {
    constexpr T object {};
    return size_t(&(object.*member)) - size_t(&object);
}

template<typename T>
concept array = requires (T t) {
    t.size();
};

template<typename T, typename M>
concept data_member = std::is_member_object_pointer_v<M T::*>;

template<typename T>
concept arithmetic = std::is_arithmetic_v<T>;



void gl_vertex_attrib(GLint attrib_index, size_t num, GLenum type, size_t stride, size_t member_offset) {
    if (type == GL_FLOAT) {
        glVertexAttribPointer(attrib_index, num, type, GL_FALSE, stride, (void*)member_offset);
    } else if (type == GL_DOUBLE) {
        glVertexAttribLPointer(attrib_index, num, type, stride, (void*)member_offset);
    } else if (
        type == GL_BYTE ||
        type == GL_UNSIGNED_BYTE ||
        type == GL_SHORT ||
        type == GL_UNSIGNED_SHORT ||
        type == GL_INT ||
        type == GL_UNSIGNED_INT
    ) {
        glVertexAttribIPointer(attrib_index, num, type, stride, (void*)member_offset);
    } else {
        throw std::runtime_error("error, unsupported type parameter to glVertexAttrib*Pointer in bind");
    }
}

template<typename T>
requires arithmetic<T>
void gl_vertex_attrib(GLint attrib_index, T t) {
    size_t num = 1;
    GLenum type = type_to_gltype<T>();
    size_t member_offset = 0;
    size_t stride = sizeof(T);

    gl_vertex_attrib(attrib_index, num, type, stride, member_offset);
}
template<typename T, typename M>
requires data_member<T, M> and arithmetic<M>
void gl_vertex_attrib(GLint attrib_index, M T::* member_pointer) {
    size_t num = 1;
    GLenum type = type_to_gltype<M>();
    size_t member_offset = offset_of<T, M>(member_pointer);
    size_t stride = sizeof(T);

    gl_vertex_attrib(attrib_index, num, type, stride, member_offset);
}
template<typename T>
requires array<T>
void gl_vertex_attrib(GLint attrib_index, T t) {
    constexpr T tmp{};
    size_t num = tmp.size();
    GLenum type = type_to_gltype<typename T::value_type>();
    size_t member_offset = 0;
    size_t stride = sizeof(T);

    gl_vertex_attrib(attrib_index, num, type, stride, member_offset);
}
template<typename T, typename M>
requires data_member<T, M> and array<M>
void gl_vertex_attrib(GLint attrib_index, M T::* member_pointer) {
    constexpr M tmp{};
    size_t num = tmp.size();
    GLenum type = type_to_gltype<typename M::value_type>();
    size_t member_offset = offset_of<T, M>(member_pointer);
    size_t stride = sizeof(T);

    gl_vertex_attrib(attrib_index, num, type, stride, member_offset);
}
