#pragma once

struct layer_t {
    virtual void draw() = 0;
    virtual ~layer_t() {}
};
