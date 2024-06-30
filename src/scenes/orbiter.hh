#include <range/v3/all.hpp>

#include "data/pds4.hh"
#include "engine/drawable.hh"
#include "util/profiler.hh"

struct orbiter {
    drawable<> drawable;
    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<float, 3>> orbiter_positions;
    orbiter(): drawable() {
        /*
        TODO
        potentially interesting fields
        <geom:Coordinate_System_Identification>
        <lro:mission_phase_name>
        <lro:orbit_number>
        <lro:spacecraft_clock_start_count>
            does this match field MET_Seconds?
        <start_date_time>
        <geom:geometry_start_time_utc>

        for figuring out spacecraft orientation
        Off_Nadir_Angle
        Emission_Angle
        */

        for (auto& table: pds4_load_all()) {
            {
                const auto lon = table.field<int32_t>("Scaled_Spacecraft_Longitude");
                const auto lat = table.field<int32_t>("Scaled_Spacecraft_Latitude");
                const auto radius = table.field<uint32_t>("Spacecraft_Radius");
                for (size_t i = 0; i < lon.size(); i++) {
                    orbiter_positions.push_back(LLR_to_XYZ(lon[i], lat[i], radius[i]));
                }
            }
            {
                for (size_t spot = 1; spot <= 5; spot++) {
                    const auto lon = table.field<int32_t>(std::format("Longitude_{}", spot));
                    const auto lat = table.field<int32_t>(std::format("Latitude_{}", spot));
                    const auto radius = table.field<int32_t>(std::format("Radius_{}", spot));
                    for (size_t i = 0; i < lon.size(); i++) {
                        vertices.push_back(LLR_to_XYZ(lon[i], lat[i], radius[i]));
                    }
                }
            }
            if (orbiter_positions.size() > 1e6) {
                break;
            }
            std::cout << ">>>> loading... positions.size() = " << orbiter_positions.size() << std::endl;
        }
        std::cout << ">>>> loaded" << std::endl;
        //TODO extend pds4_load to get all .xml files in the directory, sorted
        //and then concat/stream the data files
        //check the all the <Table_Binary> nodes exactly match?
        //concat the dat files directly?
        //should check that they all have an exact number of bytes

        drawable.vbo.data = vertices;
        drawable.vbo.bind(drawable.shader.program_vertex, "vertex");
    }
    void draw() {
        size_t i = shared.frame;
        std::array<float, 3> op = orbiter_positions[i % orbiter_positions.size()];
        glm::vec3 orbiter_pos = {op[0], op[1], op[2]};
        shared.inputs.view = glm::lookAt(
            orbiter_pos,
            glm::vec3(),
            glm::vec3(0, 0, 1)
        );
        shared.inputs.projection = glm::perspective(glm::radians(4.0f), static_cast<float>(shared.inputs.resolution_x) / shared.inputs.resolution_y, 10.0f, 2 * 1.8e6f);
        //drawable.vbo.data = vertices;
        //drawable.vbo.data.resize(i * 5);
        shared.draw();
        drawable.draw(GL_POINTS);
    }
};
