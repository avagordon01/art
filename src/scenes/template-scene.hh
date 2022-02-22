#include "engine/drawable.hh"

struct template_scene {
    drawable<> drawable;
    template_scene():
        drawable()
    {
    }
    void draw() {
        drawable.draw();
    }
};
