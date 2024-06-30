#pragma once

#include <cassert>
#include <cstddef>
#include <experimental/mdspan>
#include <filesystem>
#include <libassert/assert.hpp>
#include <memory>
#include <pugixml.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "engine/mmap.hh"

namespace stdx = std::experimental;

template <typename T>
std::string type_to_pds4_data_type_string() {
    assert(std::is_integral_v<T>);
    //assert(std::is_unsigned_v<T>);
    if constexpr (std::is_same_v<T, uint8_t>) {
        return "UnsignedByte";
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return "UnsignedLSB2";
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return "UnsignedLSB4";
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return "UnsignedLSB8";
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return "SignedByte";
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return "SignedLSB2";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "SignedLSB4";
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return "SignedLSB8";
    } else {
        static_assert(false);
    }
}

struct pds4_field_binary {
    size_t size;
    size_t offset;
    double missing_constant;
    std::string data_type;
    std::string unit;
};

using extents = stdx::dextents<size_t, 1>;
struct pds4_table_binary {
    std::unordered_map<std::string, pds4_field_binary> fields;
    size_t stride;
    size_t count;
    std::unique_ptr<mmap_file> file{};
    template <typename T>
    stdx::mdspan<T, extents, stdx::layout_stride> field(const std::string&);
};

template <typename T>
stdx::mdspan<T, extents, stdx::layout_stride> pds4_table_binary::field(const std::string& field_name) {
    const auto& field = fields.at(field_name);
    size_t size = field.size;
    ASSERT(sizeof(T) == size);
    ASSERT(type_to_pds4_data_type_string<T>() == field.data_type);
    ASSERT((stride % sizeof(T)) == 0);
    std::span<std::byte> whole_file = file->data.subspan(field.offset);
    T* ptr = reinterpret_cast<T*>(whole_file.data());
    //TODO check that the last element is in the span after this integer division
    //stdx::mdspan<T, extents> x(ptr, whole_file.size() / sizeof(T));
    extents sub_ext(count);
    std::array strides{stride / sizeof(T)};
    stdx::layout_stride::mapping sub_map(sub_ext, strides);
    return stdx::mdspan<T, extents, stdx::layout_stride>(ptr, sub_map);
}

std::vector<std::filesystem::path> all_files_with_extension_sorted_by_filename(
    std::filesystem::path directory, std::string_view extension
) {
    std::vector<std::filesystem::path> files;
    for (auto& x: std::filesystem::recursive_directory_iterator(directory)) {
        if (x.is_regular_file() && x.path().string().ends_with(extension)) {
            files.push_back(x);
        }
    }
    std::sort(files.begin(), files.end(), [](auto a, auto b) -> bool { return a.filename() < b.filename(); });
    return files;
}

pds4_table_binary pds4_load(std::filesystem::path filename) {
    using namespace std::string_view_literals;
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());
    if (!result) {
        throw std::runtime_error("pugi xml failed to load " + filename.string() + " reason: " + result.description());
    }

    std::filesystem::path data_filename = doc.child("Product_Observational")
                                              .child("File_Area_Observational")
                                              .child("File")
                                              .child("file_name")
                                              .child_value();
    data_filename = std::filesystem::path(filename).replace_filename(data_filename);

    const auto table_binary = doc.child("Product_Observational").child("File_Area_Observational").child("Table_Binary");
    const auto record_binary = table_binary.child("Record_Binary");

    ASSERT("byte"sv == table_binary.child("offset").attribute("unit").value());
    ASSERT("0"sv == table_binary.child("offset").child_value());

    pds4_table_binary tb;

    tb.count = std::stoull(table_binary.child("records").child_value());
    ASSERT("byte"sv == record_binary.child("record_length").attribute("unit").value());
    tb.stride = std::stoull(record_binary.child("record_length").child_value());
    if (std::filesystem::exists(data_filename)) {
        tb.file = std::make_unique<mmap_file>(data_filename);
        ASSERT(tb.file->data.size() == tb.stride * tb.count);
    }

    for (const auto field_binary: record_binary.children("Field_Binary")) {
        const auto name = field_binary.child("name").child_value();
        ASSERT("byte"sv == field_binary.child("field_location").attribute("unit").value());
        const auto offset = field_binary.child("field_location").text().as_ullong();
        const auto data_type = field_binary.child("data_type").child_value();
        //ASSERT(!data_type.contains("MSB"));
        ASSERT("byte"sv == field_binary.child("field_length").attribute("unit").value());
        const auto size = field_binary.child("field_length").text().as_ullong();
        const auto missing_constant =
            field_binary.child("Special_Constants").child("missing_constant").text().as_double();
        const auto unit = field_binary.child("unit").child_value();

        tb.fields.insert({
            name,
            pds4_field_binary{
                              .size = size,
                              .offset = offset,
                              .missing_constant = missing_constant,
                              .data_type = data_type,
                              .unit = unit,
                              }
        });
    }

    return tb;
}

std::vector<pds4_table_binary> pds4_load_all() {
    std::filesystem::path directory = "data/lola/lrolol_1xxx/data/lola_rdr";
    const auto xml_files = all_files_with_extension_sorted_by_filename(directory, ".xml");
    //TODO check all Record_Binary tags are the same

    std::vector<pds4_table_binary> tables;
    size_t total = 0;
    for (const auto& xml_file: xml_files) {
        auto table = pds4_load(xml_file);
        if (table.file) {
            tables.push_back(std::move(table));
        }
        total += table.count;
    }
    std::cout << ">>>> total records in dataset = " << total << std::endl;
    return tables;
}

template <typename A, typename B, typename C>
std::array<float, 3> LLR_to_XYZ(A longitude, B latitude, C radius) {
    double lon = (static_cast<double>(longitude) / 1e7) / (180 / std::numbers::pi);
    double lat = (static_cast<double>(latitude) / 1e7) / (180 / std::numbers::pi);
    double r = static_cast<double>(radius) / 1e3;
    double x = r * cos(lat) * cos(lon);
    double y = r * cos(lat) * sin(lon);
    double z = r * sin(lat);
    return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}
