#pragma once
#include <array>
#include <cstdint>

namespace rmps::huffman {

inline constexpr int NUM_TABLES = 34;

/// Lightweight description of one Huffman table. `code` and `hlen` point at
/// `count`-element arrays of code words and bit lengths respectively.
/// Empty/sparse table rows have `code == nullptr`.
struct Table {
    int xlen;
    int ylen;
    int linbits;
    int linmax;
    const std::uint16_t* code;
    const std::uint16_t* hlen;
    int count;
};

extern const std::array<Table, NUM_TABLES> tables;

inline const Table& at(int idx) { return tables[static_cast<std::size_t>(idx)]; }

} // namespace rmps::huffman
