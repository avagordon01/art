#include <bit>
#include <span>
#include <algorithm>

struct span_reader {
    std::span<std::byte> data;
    size_t file_offset_size;
    size_t machine_segment_size;
    size_t machine_address_size;

    span_reader(std::span<std::byte> data_):
        data(data_)
    {}

    void reset(std::span<std::byte> data_) {
        data = data_;
    }

    std::span<std::byte> read_bytes(size_t size) {
        auto r = data.first(size);
        data = data.subspan(size);
        return r;
    }
};

template<typename T>
T fix_endianness(T value, std::endian input_endianness) {
    if (input_endianness == std::endian::native) {
        return value;
    } else {
        //return std::byteswap(value);
        std::byte* x = reinterpret_cast<std::byte*>(&value);
        std::reverse(x, x + sizeof(value));
        return value;
    }
}

template<typename R, typename T>
requires std::is_integral_v<T>
void read(R& r, T& v) {
    v = *std::bit_cast<T*>(r.read_bytes(sizeof(v)).data());
    v = fix_endianness(v, r.input_endianness);
}

template<typename T>
requires std::is_integral_v<T>
T read_integer(std::span<std::byte> x, std::endian endianness) {
    assert(x.size() == sizeof(T));
    T v = *std::bit_cast<T*>(x.data());
    v = fix_endianness(v, endianness);
    return v;
}

template<typename T>
span_reader& operator&(span_reader &r, T& v) {
    read(r, v);
    return r;
}

template<typename A>
std::span<A> span_cast(std::span<std::byte> x) {
    return std::span<A>(
        reinterpret_cast<A*>(x.data()),
        x.size() / sizeof(A)
    );
}
