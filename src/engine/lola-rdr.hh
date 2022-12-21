#pragma once
#include <string>
#include <vector>
#include <array>
#include <algorithm>

#include "util/profiler.hh"
#include "mmap.hh"

struct lola_rdr_format {
    int32_t MET_SECONDS;
    uint32_t SUBSECONDS;

    struct {
        uint32_t SECONDS;
        uint32_t SUBSECONDS;
    } TRANSMIT_TIME;

    int32_t LASER_ENERGY; //unit NANOJOULES
    int32_t TRANSMIT_WIDTH; //unit PICOSECONDS
    int32_t SC_LONGITUDE; //unit DEGREES * (10**7)
    // -180 to 180
    int32_t SC_LATITUDE; //unit DEGREES * (10**7)
    // -90 to 90
    uint32_t SC_RADIUS; //unit MILLIMETERS
    uint32_t SELENOID_RADIUS; //unit MILLIMETERS

    struct {
        int32_t LONGITUDE; //unit DEGREES * (10**7)
        int32_t LATITUDE; //unit DEGREES * (10**7)
        // -90 to 90
        int32_t RADIUS; //unit MILLIMETERS
        uint32_t RANGE; //unit MILLIMETERS
        int32_t PULSE; //unit PICOSECOND
        uint32_t ENERGY; //unit ZEPTOJOULES
        uint32_t BACKGROUND; //unit COUNTS
        uint32_t THRESHOLD; //unit NANOVOLTS
        uint32_t GAIN;
        uint16_t SHOT_FLAG;
        uint16_t UNCERTAINTY;
    } spots[5];

    uint16_t OFFNADIR_ANGLE; //unit RADIANS * 20,000
    uint16_t EMISSION_ANGLE; //unit RADIANS * 20,000
    uint16_t SOLAR_INCIDENCE; //unit RADIANS * 20,000
    uint16_t SOLAR_PHASE; //unit RADIANS * 20,000
    uint32_t EARTH_RANGE;
    uint16_t EARTH_PULSE; //unit PICOSECOND
    uint16_t EARTH_ENERGY; //unit ATTOJOULE
};
static_assert(sizeof(lola_rdr_format) == 256);
static_assert(offsetof(lola_rdr_format, TRANSMIT_WIDTH) == 20);
static_assert(offsetof(lola_rdr_format, EARTH_ENERGY) == 254);

template<typename A>
std::span<A> span_cast(std::span<std::byte> x) {
    return std::span<A>(
        reinterpret_cast<A*>(x.data()),
        x.size() / sizeof(A)
    );
}

template<typename A, typename B, typename C>
std::array<float, 3> LLR_to_XYZ(A latitude, B longitude, C radius) {
    double lon = (static_cast<double>(longitude) / 1e7) / (180 / std::numbers::pi);
    double lat = (static_cast<double>(latitude) / 1e7) / (180 / std::numbers::pi);
    double r = static_cast<double>(radius) / 1e3;
    double x = r * cos(lat) * cos(lon);
    double y = r * cos(lat) * sin(lon);
    double z = r * sin(lat);
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z)
    };
}

struct geospatial {
    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<float, 3>> colours;
    std::vector<std::array<float, 3>> orbiter_positions;
    geospatial(const std::vector<std::string>& filenames/*, std::string short_name*/) {
        for (auto& filename: filenames) {
            mmap_file mf{filename};
            std::span<lola_rdr_format> records = span_cast<lola_rdr_format>(mf.data);
            //assert(records.size() == 190120);
            std::vector<uint32_t> energies;
            size_t j = 0;
            for (auto& record: records) {
                j++;
                if (j % 10 > 1) {
                    continue;
                }
                orbiter_positions.push_back(LLR_to_XYZ(record.SC_LATITUDE, record.SC_LONGITUDE, record.SC_RADIUS));

                for (size_t i = 0; i < 1; i++) {
                    auto& spot = record.spots[i];
                    if ((spot.SHOT_FLAG & 1) == 0) {
                        //this never seems to be hit
                        continue;
                    }
                    energies.push_back(spot.ENERGY);
                    vertices.push_back(LLR_to_XYZ(spot.LATITUDE, spot.LONGITUDE, spot.RADIUS));
                }
            }
            uint32_t min_energy = *std::ranges::min_element(energies);
            uint32_t max_energy = *std::ranges::max_element(energies);
            colours.resize(energies.size());
            std::ranges::transform(energies, colours.begin(), [&min_energy, &max_energy](uint32_t energy){
                float x = static_cast<float>(energy - min_energy) / (max_energy - min_energy);
                x = x / 2 + 0.5;
                return std::array<float, 3>{x, x, x};
            });
        }
    }
};
