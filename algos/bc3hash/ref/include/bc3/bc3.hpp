#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace bc3 {
using Digest = std::array<std::uint8_t, 32>;
using Block512 = std::array<std::uint8_t, 64>;
// Initializes shared, immutable precomputation. Optional: first use does this
// automatically. Call before starting a latency measurement or worker threads.
void initialize();
enum class Mode { compact,
                  fast };
// Select before the first hash/initialize call. Both modes have identical
// outputs; compact uses ~7 MiB of tables, fast ~257 MiB. Later changes throw.
void configure(Mode mode);
struct BuildInfo {
    unsigned curve_window_bits;
    bool gmp_inversion;
    std::size_t mix_table_bytes;
};
BuildInfo build_info() noexcept;
// Reads exactly 64 bytes. Input may be unaligned. Thread-safe; no folding.
Digest bc3mix512(const void* input64);
inline Digest bc3mix512(const Block512& input) { return bc3mix512(input.data()); }
// All one-shot functions accept nullptr only when size is zero.
Digest bc3mix(const void* data, std::size_t size);
Digest bc3hash(const void* data, std::size_t size);
Block512 bc3fold(const void* data, std::size_t size);
std::string hex(const Digest& digest);

// Incremental interfaces use bounded memory. finish() consumes the state;
// further calls to update()/finish() throw std::logic_error.
class Mix
{
public:
    Mix();
    ~Mix();
    Mix(Mix&&) noexcept;
    Mix& operator=(Mix&&) noexcept;
    Mix(const Mix&) = delete;
    Mix& operator=(const Mix&) = delete;
    void update(const void* data, std::size_t size);
    Digest finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class Hash
{
public:
    Hash();
    ~Hash();
    Hash(Hash&&) noexcept;
    Hash& operator=(Hash&&) noexcept;
    Hash(const Hash&) = delete;
    Hash& operator=(const Hash&) = delete;
    void update(const void* data, std::size_t size);
    Digest finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace bc3
