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

struct monospace_printable_ascii_font_atlas {
    size_t width_ = 0;
    size_t height_ = 0;
    GLuint texture_ = 0;

    GLuint binding_id = 1;
    std::string header_shader_text = R"foo(
layout(binding=)foo" + std::to_string(binding_id) + R"foo() uniform usampler2DArray font_atlas;
)foo";

    monospace_printable_ascii_font_atlas(std::string filename) {
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

        if (face->num_fixed_sizes > 1) {
            throw std::runtime_error("more than one size in font!");
        }
        width_ = face->available_sizes[0].width;
        height_ = face->available_sizes[0].height;

        std::vector<char> texels;
        texels.resize(colour_channels() * num_chars() * width() * height());
        for (char c = ' '; c <= '~'; c++) {
            error = FT_Load_Char(face, c, FT_LOAD_RENDER);
            if (error) {
                throw std::runtime_error("load char error!");
            }

            FT_GlyphSlot slot = face->glyph;
            FT_Bitmap* bitmap = &slot->bitmap;
            if (bitmap->pixel_mode != FT_PIXEL_MODE_MONO) {
                throw std::runtime_error("bitmap is not 1bpp monochrome!" );
            }
            if (bitmap->width != width() || bitmap->rows != height()) {
                throw std::runtime_error("bitmap is not monospaced!" );
            }
            for (unsigned y = 0; y < bitmap->rows; y++) {
                for (unsigned x = 0; x < bitmap->width; x++) {
                    bool bit = (bitmap->buffer[y * bitmap->pitch + x / 8] >> (7 - (x % 8))) & 1;
                    if (bit) {
                        std::fill_n(texels.begin() +
                            colour_channels() * ((c - ' ') * width() * height() + y * width() + x)
                        , 4, 255);
                    }
                }
            }
        }

        FT_Done_Face(face);
        FT_Done_FreeType(library);

        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D_ARRAY, texture_);
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, width(), height(), num_chars());
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, width(), height(), num_chars(), GL_RGBA, GL_UNSIGNED_BYTE, texels.data());
        glBindTextureUnit(binding_id, texture());
    }

    void draw() {
    }

    GLuint texture() {
        return texture_;
    }
    size_t colour_channels() {
        return 4;
    }
    size_t num_chars() {
        return 1 + '~' - ' ';
    }
    size_t width() {
        return width_;
    }
    size_t height() {
        return height_;
    }

};
