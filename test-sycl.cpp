#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include "data/pds4.hh"

int main() {
    size_t total = 0;
    size_t processed = 0;
    auto t0 = std::chrono::system_clock::now();
    std::vector<pds4_table_binary> tables = pds4_load_all(total);
    auto t1 = std::chrono::system_clock::now();
    sycl::queue queue;
    for (auto& table: tables) {
        size_t spots = 5;
        sycl::range<1> workspace_size{table.count};
        sycl::buffer<std::array<float, 3>, 1> positions(workspace_size);
        sycl::buffer<std::array<float, 3>, 2> vertices({table.count, spots});
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor positions_acs{positions, cgh, sycl::write_only};
            sycl::accessor vertices_acs{vertices, cgh, sycl::write_only};
            const auto sc_lon = table.field<int32_t>("Scaled_Spacecraft_Longitude");
            const auto sc_lat = table.field<int32_t>("Scaled_Spacecraft_Latitude");
            const auto sc_radius = table.field<uint32_t>("Spacecraft_Radius");
            size_t spot = 1;
            //TODO mdspan table.field_array
            const auto spot_lon = table.field<int32_t>(std::format("Longitude_{}", spot));
            const auto spot_lat = table.field<int32_t>(std::format("Latitude_{}", spot));
            const auto spot_radius = table.field<int32_t>(std::format("Radius_{}", spot));
            cgh.parallel_for<class Positions>(workspace_size, [=](sycl::id<1> workspace_id) {
                size_t i = workspace_id.get(0);
                positions_acs[i] = LLR_to_XYZ(sc_lon[i], sc_lat[i], sc_radius[i]);
                for (size_t spot = 0; spot < spots; spot++) {
                    vertices_acs[{workspace_id, spot}] = LLR_to_XYZ(spot_lon[i], spot_lat[i], spot_radius[i]);
                }
            });
        });
        processed += workspace_size.get(0);
    }
    queue.wait();
    auto t2 = std::chrono::system_clock::now();
    std::cout << "processed / total = " << processed << " / " << total << " = " << (100.0f * processed / total) << "%" << std::endl;
    std::cout << "mapping took " << std::chrono::duration<float>(t1 - t0).count() << "s" << std::endl;
    std::cout << "processing took " << std::chrono::duration<float>(t2 - t1).count() << "s" << std::endl;

    return 0;
}
