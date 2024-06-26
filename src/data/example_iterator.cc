#include <span>
#include <ranges>
#include <range/v3/all.hpp>

struct example_cursor {
    decltype(auto) read() const {
        return data[index];
    }

    bool equal(const example_cursor& other) const {
        return index == other.index;
    }

    void advance(std::ptrdiff_t n) {
        index += n;
    }
    void next() {
        advance(1);
    }
    void prev() {
        advance(-1);
    }
    std::ptrdiff_t distance_to(const example_cursor& other) const {
        return other.index - this->index;
    }

    example_cursor() = default;
    explicit example_cursor(int index_, std::span<int> data_):
        index(index_),
        data(data_)
    {}

    int index;
    std::span<int> data;
};
using example_iterator = ranges::basic_iterator<example_cursor>;
static_assert(std::random_access_iterator<example_iterator>);
