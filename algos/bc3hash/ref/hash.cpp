#include "standard_hashes.hpp"
#include <bc3/bc3.hpp>

#include <stdexcept>
#include <string_view>

namespace bc3 {
namespace {
constexpr std::string_view domain = "BC3Hash-BlockHash-Algorithm/BitcoinIII-PoW/2026";
// Derived by the exact hashlib formulas in the authoritative bc3hash.py.
constexpr std::array<std::uint8_t, 8> person_s{{0x0a, 0x20, 0xe8, 0x11, 0x61, 0x09, 0x08, 0xba}};
constexpr std::array<std::uint8_t, 16> person_b{{0x1e, 0x80, 0xf2, 0xfd, 0xdc, 0x58, 0x18, 0xd6, 0xfe, 0x9b, 0x58, 0x1e, 0x3b, 0xd4, 0x93, 0x0b}};
template <class H>
void tag(H& h, std::string_view suffix)
{
    h.update(domain.data(), domain.size());
    h.update(suffix.data(), suffix.size());
}
struct State {
    standard::Sha3 sha3_512{512}, sha3_384{384};
    standard::Sha512 sha512;
    standard::Ripemd160 ripemd;
    standard::Blake2s blake_s{person_s};
    standard::Blake2b blake_b{person_b};
    State()
    {
        tag(sha3_512, "/SHA3-512");
        tag(sha3_384, "/SHA3-384");
        tag(sha512, "/SHA-512");
        tag(ripemd, "/RIPEMD-160");
        tag(blake_s, "/BLAKE2s");
        tag(blake_b, "/BLAKE2b");
    }
    void update(const void* data, std::size_t n)
    {
        if (n && !data) throw std::invalid_argument("Null BC3 input");
        sha3_512.update(data, n);
        sha3_384.update(data, n);
        sha512.update(data, n);
        ripemd.update(data, n);
        blake_s.update(data, n);
        blake_b.update(data, n);
    }
    Digest finish()
    {
        Block512 preimage;
        sha3_512.finish(preimage.data());
        const auto mix = bc3mix512(preimage);
        std::array<std::uint8_t, 48> s384;
        sha3_384.finish(s384.data());
        const auto s512 = sha512.finish();
        const auto r160 = ripemd.finish();
        const auto bs = blake_s.finish();
        const auto bb = blake_b.finish();
        standard::Sha3 final{256};
        tag(final, "/SHA3-256");
        final.update(s384.data(), s384.size());
        final.update(mix.data(), mix.size());
        final.update(s512.data(), s512.size());
        final.update(r160.data(), r160.size());
        final.update(bs.data(), bs.size());
        final.update(bb.data(), bb.size());
        Digest output;
        final.finish(output.data());
        return output;
    }
};
} // namespace
struct Hash::Impl {
    State state;
    bool done = false;
};
Hash::Hash() : impl_(std::make_unique<Impl>()) {}
Hash::~Hash() = default;
Hash::Hash(Hash&&) noexcept = default;
Hash& Hash::operator=(Hash&&) noexcept = default;
void Hash::update(const void* data, std::size_t n)
{
    if (!impl_ || impl_->done) throw std::logic_error("BC3Hash state already finished or moved");
    impl_->state.update(data, n);
}
Digest Hash::finish()
{
    if (!impl_ || impl_->done) throw std::logic_error("BC3Hash state already finished or moved");
    impl_->done = true;
    return impl_->state.finish();
}
Digest bc3hash(const void* data, std::size_t n)
{
    // No allocation in the one-shot consensus path after shared initialization.
    State state;
    state.update(data, n);
    return state.finish();
}
} // namespace bc3
