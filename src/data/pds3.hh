#include <algorithm>
#include <bit>
#include <complex>
#include <filesystem>
#include <iterator>
#include <llnl-units/units.hpp>
#include <range/v3/all.hpp>
#include <ranges>
#include <span>
#include <typeinfo>
#include <variant>

#include "engine/mmap.hh"
#include "interleave.hpp"

std::string strip_whitespace(std::string x) {
    x.erase(std::remove_if(x.begin(), x.end(), [](unsigned char x) { return std::isspace(x); }), x.end());
    return x;
}
std::string strip_commas(std::string x) {
    x.erase(std::remove(x.begin(), x.end(), ','), x.end());
    return x;
}
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

struct pds3_column_format {
    size_t start_byte;
    size_t bytes;
    size_t stride;
    std::endian endianness;
    std::variant<
        int8_t, int16_t, int32_t, uint8_t, uint16_t, uint32_t, float, double, long double, std::complex<float>,
        std::complex<double>, std::complex<long double>>
        data_type;
    units::precise_unit unit;
};
std::unordered_map<std::string, pds3_column_format> pds3_fmt_parse(std::string fmt_path) {
    std::unordered_map<std::string, pds3_column_format> columns;

    std::ifstream ifs(fmt_path, std::ios::in);
    if (!ifs) {
        throw std::runtime_error("couldn't open file " + fmt_path);
    }

    pds3_column_format column{};
    std::string name = "";
    size_t record_size = 0;
    std::string data_type = "";
    while (!ifs.eof()) {
        std::string line;
        std::getline(ifs, line, '\n');
        size_t p = line.find('=');
        if (p == std::string::npos) {
            continue;
        }
        std::string key = strip_whitespace(line.substr(0, p));
        std::string value = strip_whitespace(line.substr(p + 1));
        if (key.empty() || value.empty()) {
            continue;
        }

        if (key == "NAME") {
            name = value;
        } else if (key == "START_BYTE") {
            column.start_byte = stoll(value) - 1;
        } else if (key == "BYTES") {
            column.bytes = stoll(value);
            record_size += column.bytes;
        } else if (key == "UNIT") {
            std::string unit = strip_commas(value);

            {
                /*
                for (std::string s: {
                    "DEGREES*(10**7)",
                    "RADIANS*20000",
                    "MILLIMETERS"
                }) {
                    std::cout << s << std::endl;
                    s = to_lower(s);
                    auto m = units::measurement_from_string(s, units::case_insensitive);
                    std::cout << m.as_unit().multiplier() << std::endl;
                    std::cout << m.units().multiplier() << std::endl;
                    std::cout << m.value_as(units::default_unit("angle")) << std::endl;
                    std::cout << m.value_as(units::default_unit("length")) << std::endl;
                    auto u = units::unit_from_string(s, units::case_insensitive);
                    std::cout << u.multiplier() << std::endl;
                }
                */
                for (std::string s: {"DEGREES*(10**7)", "RADIANS*20000", "MILLIMETERS"}) {
                    std::string measurement = s;
                    std::string toUnits = "degrees*1000000";
                    auto meas = units::measurement_from_string(measurement);
                    units::precise_unit u2;
                    if (toUnits == "*" || toUnits == "<base>") {
                        u2 = meas.convert_to_base().units();
                        toUnits = units::to_string(u2);
                    } else {
                        //u2 = units::default_unit("angle");
                        u2 = units::unit_from_string(toUnits);
                    }
                    std::cout << meas.value_as(u2) << std::endl;
                }
                //FIXME
            }

            std::cout << unit << std::endl;
            column.unit = units::unit_from_string(unit, units::case_insensitive);
            std::cout << units::to_string(column.unit) << std::endl;
        } else if (key == "DATA_TYPE") {
            if (value.starts_with("MSB_")) {
                column.endianness = std::endian::big;
                value = value.substr(4);
            } else if (value.starts_with("LSB_")) {
                column.endianness = std::endian::little;
                value = value.substr(4);
            } else {
                column.endianness = std::endian::big;
            }
            data_type = value;
        } else if (key == "END_OBJECT") {
            if (data_type == "INTEGER") {
                if (column.bytes == 1) {
                    column.data_type = int8_t{};
                } else if (column.bytes == 2) {
                    column.data_type = int16_t{};
                } else if (column.bytes == 4) {
                    column.data_type = int32_t{};
                }
            } else if (data_type == "UNSIGNED_INTEGER") {
                if (column.bytes == 1) {
                    column.data_type = uint8_t{};
                } else if (column.bytes == 2) {
                    column.data_type = uint16_t{};
                } else if (column.bytes == 4) {
                    column.data_type = uint32_t{};
                }
            } else if (data_type == "IEEE_REAL") {
                if (column.bytes == 4) {
                    column.data_type = float{};
                } else if (column.bytes == 8) {
                    column.data_type = double{};
                } else if (column.bytes == 10) {
                    column.data_type = (long double)0;
                }
            } else if (data_type == "IEEE_COMPLEX") {
                if (column.bytes == 8) {
                    column.data_type = std::complex<float>{};
                } else if (column.bytes == 16) {
                    column.data_type = std::complex<double>{};
                } else if (column.bytes == 20) {
                    column.data_type = std::complex<long double>{};
                }
            } else {
                throw std::runtime_error("unimplemented pds3 data type " + data_type);
            }
            columns[name] = column;
            column = {};
            name = "";
            data_type = "";
        }
    }
    ifs.close();
    for (auto& [_, column]: columns) {
        column.stride = record_size;
    }
    return columns;
}

template <typename T>
struct pds3_column_cursor {
    struct mixin;

    T& read() const {
        //TODO fix endianness
        return *reinterpret_cast<T*>(_ptr);
    }
    bool equal(const pds3_column_cursor& other) const { return _ptr == other._ptr; }
    void advance(std::ptrdiff_t n) { _ptr += n * _stride; }
    void next() { advance(1); }
    void prev() { advance(-1); }
    std::ptrdiff_t distance_to(const pds3_column_cursor& other) const { return (other._ptr - this->_ptr) / _stride; }

    pds3_column_cursor() noexcept = default;
    pds3_column_cursor(size_t stride, std::byte* ptr): _stride(stride), _ptr(ptr) {}

    size_t _stride;
    std::byte* _ptr;
};
template <typename T>
struct pds3_column_cursor<T>::mixin: ranges::basic_mixin<pds3_column_cursor<T>> {
    using ranges::basic_mixin<pds3_column_cursor<T>>::basic_mixin;

    mixin(size_t stride, std::byte* ptr): mixin{pds3_column_cursor(stride, ptr)} {}
};

template <typename T>
using pds3_column_iterator = ranges::basic_iterator<pds3_column_cursor<T>>;
static_assert(std::random_access_iterator<pds3_column_iterator<uint32_t>>);

template <typename T>
struct pds3_column_container {
    pds3_column_format& column;
    std::span<std::byte> data_file;
    pds3_column_container(pds3_column_format& _column, std::span<std::byte> _data_file):
        column(_column), data_file(_data_file) {
        assert(std::holds_alternative<T>(column.data_type));
        if (data_file.size() % column.stride != 0) {
            std::cout << "warning, data file size is not an exact multiple of the record format size" << std::endl;
            std::cout << "attempting to continue by dropping end bytes" << std::endl;
            size_t to_drop = data_file.size() % column.stride;
            data_file = data_file.subspan(0, data_file.size() - to_drop);
        }
    }
    pds3_column_iterator<T> begin() {
        return pds3_column_iterator<T>(column.stride, data_file.data() + column.start_byte);
    }
    pds3_column_iterator<T> end() {
        return pds3_column_iterator<T>(column.stride, data_file.data() + data_file.size() + column.start_byte);
    }
    size_t size() { return data_file.size() / column.stride; }
};
static_assert(std::ranges::sized_range<pds3_column_container<int32_t>>);
static_assert(std::ranges::random_access_range<pds3_column_container<int32_t>>);

template <typename A, typename B, typename C>
std::array<float, 3> LLR_to_XYZ(A longitude, B latitude, C radius);

std::vector<std::array<float, 3>> pds3_load() {
    std::string data_directory = "data/lola/pds-geosciences.wustl.edu";
    std::string fmt_path = "";
    for (auto& x: std::filesystem::recursive_directory_iterator(data_directory)) {
        if (x.is_regular_file() && x.path().string().ends_with("lolardr.fmt")) {
            fmt_path = x.path().string();
        }
    }
    if (fmt_path.empty()) {
        throw std::runtime_error("no label (.fmt) file found");
    }

    auto columns = pds3_fmt_parse(fmt_path);

    std::vector<std::string> data_files;
    for (auto& x: std::filesystem::recursive_directory_iterator(data_directory)) {
        if (x.is_regular_file() && x.path().string().ends_with(".dat")) {
            data_files.push_back(x.path().string());
        }
    }
    std::sort(data_files.begin(), data_files.end());
    data_files = {
        //"data/lola/pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_es_110/concatenated.dat"
        "data/lola/pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_es_110/"
        "lolardr_213490400.dat"
    };
    std::cout << "num data files found: " << data_files.size() << std::endl;

    /*
    lbl file interesting fields
        FILE_RECORDS * RECORD_BYTES == ROWS * ROW_BYTES
        COORDINATE_SYSTEM_NAME
        MISSION_PHASE_NAME
        ORBIT_NUMBER
        START_TIME
        STOP_TIME
        SPACECRAFT_CLOCK_START_COUNT
        SPACECRAFT_CLOCK_STOP_COUNT
    */

    std::vector<std::array<float, 3>> xyz_positions;
    for (auto& data_file: data_files) {
        mmap_file mf{data_file};
        auto a = pds3_column_container<int32_t>{columns["SC_LONGITUDE"], mf.data};
        auto b = pds3_column_container<int32_t>{columns["SC_LATITUDE"], mf.data};
        auto c = pds3_column_container<uint32_t>{columns["SC_RADIUS"], mf.data};
        for (auto [sc_longitude, sc_latitude, sc_radius]: ranges::views::zip(a, b, c)) {
            LLR_to_XYZ(sc_longitude, sc_latitude, sc_radius);
        }

        size_t n = pds3_column_container<int32_t>{columns["MET_SECONDS"], mf.data}.size();
        for (size_t spot = 1; spot <= 5; spot++) {
            auto longitude =
                pds3_column_container<int32_t>{columns["LONGITUDE_" + std::to_string(spot)], mf.data}.begin();
            auto latitude =
                pds3_column_container<int32_t>{columns["LATITUDE_" + std::to_string(spot)], mf.data}.begin();
            auto radius = pds3_column_container<int32_t>{columns["RADIUS_" + std::to_string(spot)], mf.data}.begin();
            std::cout << columns["LONGITUDE_0"].unit.multiplier() << std::endl;
            std::cout << columns["LATITUDE_0"].unit.multiplier() << std::endl;
            std::cout << columns["RADIUS_0"].unit.multiplier() << std::endl;
            //FIXME
            for (size_t i = 0; i < n; i++) {
                if (i % 10 > 2) {
                    continue;
                }
                xyz_positions.push_back(LLR_to_XYZ(longitude[i], latitude[i], radius[i]));
            }
        }
        std::cout << "xyz_positions.size(): " << xyz_positions.size() << std::endl;
        std::cout << "sizeof(xyz_positions::value_type): " << sizeof(decltype(xyz_positions)::value_type) << std::endl;
    }
    return xyz_positions;

    /*
    std::vector<int> spots = {1, 2, 3, 4, 5};
    std::cerr << "good 1" << std::endl;
    std::cerr << "mf: " << mf.data.data() << ", " << mf.data.data() + mf.data.size() << std::endl;
    std::vector<std::tuple<int32_t, int32_t, uint32_t>> all_spots_all_fields = spots |
        ranges::views::transform([&](int spot) {
            auto longitude = pds3_column_container<int32_t>{columns["LONGITUDE_" + std::to_string(spot)], mf.data};
            auto latitude = pds3_column_container<int32_t>{columns["LATITUDE_" + std::to_string(spot)], mf.data};
            auto radius = pds3_column_container<int32_t>{columns["RADIUS_" + std::to_string(spot)], mf.data};
            return ranges::views::zip(longitude, latitude, radius);
        }) |
        ranges::views::take(10) |
        ranges::views::join |
        //interleave() |
        ranges::to<std::vector<std::tuple<int32_t, int32_t, uint32_t>>>();
    std::cerr << "good 2" << std::endl;
    for (auto [a, b, c]: all_spots_all_fields) {
        std::cerr << a << ", " << b << ", " << c << std::endl;
    }
    std::cerr << "good 3" << std::endl;
    exit(0);
    ranges::views::iota(1, 6) |
        ranges::views::transform([&](int spot){
            std::cerr << spot << std::endl;
            auto longitude = pds3_column_container<int32_t>{columns["LONGITUDE" + std::to_string(spot)], mf.data};
            auto latitude = pds3_column_container<int32_t>{columns["LATITUDE" + std::to_string(spot)], mf.data};
            auto radius = pds3_column_container<uint32_t>{columns["RADIUS" + std::to_string(spot)], mf.data};
            return ranges::views::zip(longitude, latitude, radius);
        }) |
        interleave_view() |
        ranges::views::for_each([&](auto tied){
            auto [longitude, latitude, radius] = tied;
            LLR_to_XYZ(longitude, latitude, radius);
        });
    */
}
