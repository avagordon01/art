#pragma once
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <string>

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <freetype/freetype.h>
#include <GL/gl.h>

#include "buffers.hh"

struct monospace_printable_unicode_font_atlas {
    size_t width_ = 0;
    size_t height_ = 0;
    storage_buffer<uint32_t> buffer;

    std::string header_shader_text = R"foo(
layout(std430) buffer font_atlas {
    uint pixels[];
};
bool bitmap_font_atlas_fetch(ivec3 tc) {
    return bitfieldExtract(pixels[(tc.z * 16) + tc.y], tc.x, 1) != 0;
};
)foo";

    monospace_printable_unicode_font_atlas(std::string filename):
        buffer({}, GL_DYNAMIC_DRAW)
    {
        FT_Error error;
        FT_Library library;
        error = FT_Init_FreeType(&library);
        if (error != 0) {
            throw std::runtime_error("init freetype error!");
        }

        FT_Face face;
        error = FT_New_Face(library, filename.c_str(), 0, &face);
        if (error != 0) {
            throw std::runtime_error("new face error!");
        }

        if (face->num_fixed_sizes == 0) {
            throw std::runtime_error("no fixed sizes in font!");
        }
        if (face->num_fixed_sizes > 1) {
            throw std::runtime_error("more than one size in font!");
        }
        width_ = 16;
        height_ = 16;
        std::cout << "width = " << width_ << ", height = " << height_ << std::endl;
        if (width_ > 32) {
            throw std::runtime_error("bitmap font width " + std::to_string(width_) + " (> 32) not supported!");
        }

        buffer.data.resize(max_chars() * height());
        size_t num_errors = 0;
        for (uint32_t c = 0; c < max_chars(); c++) {
            error = FT_Load_Char(face, c, FT_LOAD_DEFAULT);
            if (error) {
                num_errors++;
                continue;
            }

            FT_GlyphSlot slot = face->glyph;
            FT_Bitmap* bitmap = &slot->bitmap;
            if (bitmap->pixel_mode != FT_PIXEL_MODE_MONO) {
                throw std::runtime_error("bitmap is not 1bpp monochrome!" );
            }
            if (bitmap->width > width() || bitmap->rows > height()) {
                std::cout << "glyph width = " << bitmap->width << ", glyph height = " << bitmap->rows << std::endl;
                throw std::runtime_error("bitmap is not monospaced!" );
            }
            for (unsigned y = 0; y < bitmap->rows; y++) {
                for (unsigned x = 0; x < bitmap->width; x++) {
                    bool bit = (bitmap->buffer[y * bitmap->pitch + x / 8] >> (7 - (x % 8))) & 1;
                    buffer.data[c * height() + y] |= bit << x;
                }
            }
        }
        std::cout << "ignored " << num_errors << " FT_Load_Char errors" << std::endl;

        FT_Done_Face(face);
        FT_Done_FreeType(library);

        buffer.draw();
    }

    void bind(GLuint program) {
        buffer.bind(program, "font_atlas");
    }

    size_t max_chars() {
        return 0x110000;
    }
    size_t width() {
        return width_;
    }
    size_t height() {
        return height_;
    }

};
