#include "engine/drawable.hh"
#include "data/lola-rdr.hh"

#include "util/profiler.hh"

struct orbiter {
    drawable<> drawable;
    geospatial gs;
    orbiter():
        drawable(),
        gs{{
            //"data/lola/pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_co/lolardr_092581959.dat",
            //"data/lola/pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_es_110/concatenated.dat"
            "data/lola/pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_es_110/lolardr_213490400.dat"
        }}
    {
        std::vector<std::array<float, 3>> n = pds_load();
        //TODO replace gs with n

        std::cout << "num vertices " << gs.vertices.size() << std::endl;
        drawable.vbo.data = gs.vertices;
        drawable.vbo.bind(drawable.shader.program_vertex, "vertex");
    }
    void draw() {
        size_t i = shared.frame;
        std::array<float, 3> op = gs.orbiter_positions[i % gs.orbiter_positions.size()];
        glm::vec3 orbiter_pos = {op[0], op[1], op[2]};
        shared.inputs.view = glm::lookAt(
            orbiter_pos,
            glm::vec3(),
            glm::vec3(0, 0, 1)
        );
        //drawable.vbo.data = gs.vertices;
        //drawable.vbo.data.resize(i * 5);
        shared.draw();
        drawable.draw(GL_POINTS);
    }
};
