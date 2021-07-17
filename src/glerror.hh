#include <unordered_map>
#include <string>

std::unordered_map<GLenum, std::string> map_glenum_string = {
    {GL_DEBUG_TYPE_ERROR, "GL_DEBUG_TYPE_ERROR"},
    {GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, "GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR"},
    {GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, "GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR"},
    {GL_DEBUG_TYPE_PORTABILITY, "GL_DEBUG_TYPE_PORTABILITY"},
    {GL_DEBUG_TYPE_PERFORMANCE, "GL_DEBUG_TYPE_PERFORMANCE"},
    {GL_DEBUG_TYPE_MARKER, "GL_DEBUG_TYPE_MARKER"},
    {GL_DEBUG_TYPE_PUSH_GROUP, "GL_DEBUG_TYPE_PUSH_GROUP"},
    {GL_DEBUG_TYPE_POP_GROUP, "GL_DEBUG_TYPE_POP_GROUP"},
    {GL_DEBUG_TYPE_OTHER, "GL_DEBUG_TYPE_OTHER"},
    {GL_DEBUG_SEVERITY_LOW, "GL_DEBUG_SEVERITY_LOW"},
    {GL_DEBUG_SEVERITY_MEDIUM, "GL_DEBUG_SEVERITY_MEDIUM"},
    {GL_DEBUG_SEVERITY_HIGH, "GL_DEBUG_SEVERITY_HIGH"},
    {GL_DEBUG_SEVERITY_NOTIFICATION, "GL_DEBUG_SEVERITY_NOTIFICATION"},
};
std::string to_string(GLenum e) {
    auto it = map_glenum_string.find(e);
    if (it != map_glenum_string.end()) {
        return it->second;
    } else {
        return "unknown GLenum: " + std::to_string(e);
    }
}

void GLAPIENTRY gldebug_callback(
    GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam
) {
    std::cerr << "gldebug error:" << " " << to_string(type) << " " << to_string(severity) << std::endl;
    std::cerr << "\t" << message << std::endl;
}

void gl_setup_errors() {
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(gldebug_callback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
}
