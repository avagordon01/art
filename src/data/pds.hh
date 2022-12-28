#include <bit>
#include <span>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <ranges>
#include <range/v3/all.hpp>

struct record_offsets {
    size_t orbiter_latitude;
    size_t orbiter_longitude;
    size_t orbiter_radius;
    struct spot {
        size_t latitude;
        size_t longitude;
        size_t radius;
        size_t energy;
    } spots[5];
};

void strip_whitespace(std::string& x) {
    x.erase(std::remove_if(x.begin(), x.end(), [](unsigned char x){return std::isspace(x);}), x.end());
}

struct pds_fmt {
    struct pds_column {
        size_t start_byte;
        size_t bytes;
        std::string data_type;
        std::string unit;
    };
    size_t record_size = 0;
    std::unordered_map<std::string, pds_column> columns;

    pds_fmt(std::string fmt_path) {
        std::ifstream ifs(fmt_path, std::ios::in);
        if (!ifs) {
            throw std::runtime_error("couldn't open file " + fmt_path);
        }

        pds_column column {};
        std::string name = "";
        while (!ifs.eof()) {
            std::string line;
            std::getline(ifs, line, '\n');
            size_t p = line.find('=');
            if (p == std::string::npos) {
                continue;
            }
            std::string key = line.substr(0, p);
            std::string value = line.substr(p + 1);
            strip_whitespace(key);
            strip_whitespace(value);
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
            } else if (key == "DATA_TYPE") {
                column.data_type = value;
            } else if (key == "END_OBJECT") {
                columns[name] = column;
                column = {};
                name = "";
            }
        }
        ifs.close();
    }
};

template<typename T>
struct column_iterator {
    using value_type = T;
    using pointer = std::byte*;
    using reference = T&;
    using difference_type = ptrdiff_t;

    std::span<std::byte> data_file;
    pds_fmt::pds_column column;
    size_t record_size;
    pointer _ptr;

public:
    column_iterator(pds_fmt& fmt, std::span<std::byte> _data_file, std::string column_name):
        data_file(_data_file),
        column(fmt.columns[column_name]),
        record_size(fmt.record_size),
        _ptr(data_file.data() + column.start_byte)
    {
        //TODO check T matches DATA_TYPE and BYTES
        assert(sizeof(T) == column.bytes);
        if (column.data_type == "LSB_INTEGER" || column.data_type == "MSB_INTEGER") {
            assert(std::numeric_limits<T>::is_signed);
        } else if (column.data_type == "LSB_UNSIGNED_INTEGER" || column.data_type == "MSB_UNSIGNED_INTEGER") {
            assert(!std::numeric_limits<T>::is_signed);
        } else {
        }

        //TODO use UNIT
    }
    column_iterator(): _ptr(nullptr) {}
    column_iterator(pointer rhs): _ptr(rhs) {}
    column_iterator(const column_iterator &rhs): _ptr(rhs._ptr) {}

    column_iterator begin() {return column_iterator(data_file.data() + column.start_byte);}
    column_iterator end() {return column_iterator(data_file.data() + column.start_byte + data_file.size());}

    column_iterator& operator+=(difference_type rhs) {_ptr += rhs * record_size; return *this;}
    column_iterator& operator-=(difference_type rhs) {_ptr -= rhs * record_size; return *this;}
    reference operator*() const {return *reinterpret_cast<T*>(_ptr);} //TODO fix endianness?
    pointer operator->() const {return _ptr;}
    reference operator[](difference_type rhs) const {return _ptr[rhs * record_size];}

    column_iterator& operator++() {_ptr += record_size; return *this;}
    column_iterator& operator--() {_ptr -= record_size; return *this;}
    column_iterator operator++(int) {column_iterator tmp(*this); _ptr += record_size; return tmp;}
    column_iterator operator--(int) {column_iterator tmp(*this); _ptr -= record_size; return tmp;}
    difference_type operator-(const column_iterator& rhs) const {return (_ptr - rhs._ptr) / record_size;}
    column_iterator operator+(difference_type rhs) const {return column_iterator(_ptr + rhs * record_size);}
    column_iterator operator-(difference_type rhs) const {return column_iterator(_ptr - rhs * record_size);}
    friend column_iterator operator+(difference_type lhs, const column_iterator& rhs) {return column_iterator(lhs * rhs.record_size + rhs._ptr);}
    friend column_iterator operator-(difference_type lhs, const column_iterator& rhs) {return column_iterator(lhs * rhs.record_size - rhs._ptr);}

    bool operator==(const column_iterator& rhs) const {return _ptr == rhs._ptr;}
    bool operator!=(const column_iterator& rhs) const {return _ptr != rhs._ptr;}
    bool operator>(const column_iterator& rhs) const {return _ptr > rhs._ptr;}
    bool operator<(const column_iterator& rhs) const {return _ptr < rhs._ptr;}
    bool operator>=(const column_iterator& rhs) const {return _ptr >= rhs._ptr;}
    bool operator<=(const column_iterator& rhs) const {return _ptr <= rhs._ptr;}
};
static_assert(std::random_access_iterator<column_iterator<uint32_t>>);

template <typename T>
T read_column(std::span<std::byte> record, pds_fmt::pds_column& column, std::endian endianness) {
    return read_integer<T>(record.subspan(column.start_byte, column.bytes), endianness);
}

template<typename A, typename B, typename C>
std::array<float, 3> LLR_to_XYZ(A latitude, B longitude, C radius);

void pds_load() {
    std::string fmt_path = "";
    for (auto& x: std::filesystem::recursive_directory_iterator("data/lola/pds-geosciences.wustl.edu")) {
        if (x.is_regular_file() && x.path().string().ends_with("lolardr.fmt")) {
            fmt_path = x.path().string();
        }
    }
    if (fmt_path.empty()) {
        throw std::runtime_error("no label (.fmt) file found");
    }

    pds_fmt fmt(fmt_path);

    std::vector<std::string> data_files;
    for (auto& x: std::filesystem::recursive_directory_iterator("data/lola/pds-geosciences.wustl.edu")) {
        if (x.is_regular_file() && x.path().string().ends_with(".dat")) {
            data_files.push_back(x.path().string());
        }
    }
    std::sort(data_files.begin(), data_files.end());
    data_files = {"data/lola/pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_es_110/concatenated.dat"};

    for (auto& data_file: data_files) {
        mmap_file mf{data_file};
        auto it_longitude = column_iterator<int32_t>(fmt, mf.data, "SC_LONGITUDE");
        auto it_latitude = column_iterator<int32_t>(fmt, mf.data, "SC_LATITUDE");
        auto it_radius = column_iterator<uint32_t>(fmt, mf.data, "SC_RADIUS");
        for (auto [longitude, latitude, radius]: ranges::views::zip(it_longitude, it_latitude, it_radius)) {
            std::cout << longitude << ", " << latitude << ", " << radius << std::endl;
            LLR_to_XYZ(longitude, latitude, radius);
        }
    }
}
