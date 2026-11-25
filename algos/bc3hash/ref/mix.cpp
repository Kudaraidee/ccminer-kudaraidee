#include "constants.hpp"
#include "mix_internal.hpp"
#include <bc3/bc3.hpp>
#include <secp256k1.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace bc3::detail {
using namespace constants;
namespace {
std::mutex config_mutex;
std::atomic<unsigned> configured_window{10};
bool configuration_locked = false;
unsigned begin_initialization()
{
    std::lock_guard<std::mutex> lock(config_mutex);
    configuration_locked = true;
    return configured_window.load(std::memory_order_relaxed);
}
unsigned active_window()
{
    static const unsigned w = begin_initialization();
    return w;
}
constexpr FixedExponent<M64, 3> exp64(E64);
constexpr FixedExponent<M128, 4> exp128(E128);
constexpr FixedExponent<M256, 4> exp256(E256);
constexpr FixedExponent<M512, 5> exp512(E512);
struct SecpContext {
    secp256k1_context* p = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    SecpContext()
    {
        if (!p) throw std::bad_alloc();
    }
    ~SecpContext() { secp256k1_context_destroy(p); }
};
const SecpContext& context()
{
    static const SecpContext c;
    return c;
}
struct alignas(64) CurveTable {
    const unsigned window = active_window();
    const unsigned digits = (256 + window - 1) / window;
    const unsigned entries = (1u << window) - 1;
    const unsigned last_entries = (1u << (256 - window * (digits - 1))) - 1;
    // Each opaque public key occupies one cache line. The final window has
    // fewer bits; omit unreachable entries instead of padding that row.
    struct alignas(64) Key {
        secp256k1_pubkey value;
    };
    std::vector<Key> table;
    CurveTable(U256 x, U256 y) : table((digits - 1) * entries + last_entries)
    {
        const auto* ctx = context().p;
        std::array<unsigned char, 65> encoded{};
        encoded[0] = 4;
        x.store(encoded.data() + 1);
        y.store(encoded.data() + 33);
        secp256k1_pubkey base;
        if (!secp256k1_ec_pubkey_parse(ctx, &base, encoded.data(), 65)) throw std::runtime_error("Invalid BC3 curve constant");
        for (unsigned i = 0; i < digits; ++i) {
            auto* row = table.data() + i * entries;
            const unsigned count = i + 1 == digits ? last_entries : entries;
            row[0].value = base;
            for (unsigned j = 1; j < count; ++j) {
                const secp256k1_pubkey* points[] = {&row[j - 1].value, &base};
                if (!secp256k1_ec_pubkey_combine(ctx, &row[j].value, points, 2)) throw std::runtime_error("BC3 curve table failure");
            }
            if (i + 1 < digits) {
                const secp256k1_pubkey* points[] = {&row[entries - 1].value, &row[0].value};
                if (!secp256k1_ec_pubkey_combine(ctx, &base, points, 2)) throw std::runtime_error("BC3 curve table failure");
            }
        }
    }
    std::pair<U256, U256> operator()(U256 k) const
    {
        constexpr auto order = U256::hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
        if (k >= order) k = k - order;
        if (k.zero()) return {INVALID_POINT_x, INVALID_POINT_y};
        std::array<const secp256k1_pubkey*, 26> points;
        std::size_t n = 0;
        for (unsigned i = 0; i < digits; ++i) {
            const unsigned pos = i * window, limb = pos / 64, shift = pos % 64;
            std::uint64_t b = k.v[limb] >> shift;
            if (shift && limb + 1 < 4) b |= k.v[limb + 1] << (64 - shift);
            b &= entries;
            if (b) {
                points[n] = &table[i * entries + b - 1].value;
                // Issue independent cache misses together, ahead of the
                // library's dependent sequence of elliptic-curve additions.
                bc3_prefetch(points[n]);
                ++n;
            }
        }
        secp256k1_pubkey result;
        if (!secp256k1_ec_pubkey_combine(context().p, &result, points.data(), n)) throw std::runtime_error("BC3 point combination failed");
        std::array<unsigned char, 65> encoded;
        std::size_t length = encoded.size();
        if (!secp256k1_ec_pubkey_serialize(context().p, encoded.data(), &length, &result, SECP256K1_EC_UNCOMPRESSED)) throw std::runtime_error("BC3 point serialization failed");
        return {U256::load(encoded.data() + 1), U256::load(encoded.data() + 33)};
    }
};
struct Tables {
    FixedPower<F64> pow64{g64};
    FixedPower<F128> pow128{g128};
    FixedPower<F256> pow256{g256};
    std::array<CurveTable, 4> curve{{{H_x, H_y}, {J_x, J_y}, {K_x, K_y}, {L_x, L_y}}};
};
const Tables& tables()
{
    static const Tables t;
    return t;
}
const FixedPower<F512>& fold_power()
{
    static const FixedPower<F512> p{g512};
    return p;
}
const CurveTable& generator_table()
{
    static const CurveTable t{
        U256::hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"),
        U256::hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8")};
    return t;
}
} // namespace
// ccminer oracle: Fermat inversion replaces the coin's binary Euclid, which
// needs signed 128-bit integers (see vendor.py). Same unique result, including
// inverse256(0) == 0.
U256 inverse256(U256 a)
{
    a = F256::normalize(a);
    if (a.zero()) return {};
    return F256::pow(a, F256::modulus() - U256(2));
}
std::pair<U256, U256> curve_multiply(unsigned point, U256 scalar)
{
    if (point > 4) throw std::invalid_argument("BC3 curve point index");
    return point == 4 ? generator_table()(scalar) : tables().curve[point](scalar);
}
static Digest mix512(const void* input, unsigned folded)
{
    const auto& t = tables();
    const auto* bytes = static_cast<const std::uint8_t*>(input);
    std::array<U64, 32> m64;
    std::array<U128, 48> m128;
    std::array<U256, 64> m256;
    for (unsigned i = 0; i < 8; ++i)
        m64[i] = U64::load(bytes + 8 * i);
    auto c1 = m64[0] ^ base[0], c2 = m64[1] ^ base[1], c3 = m64[2] ^ base[2], c4 = m64[3] ^ base[3];
    auto c5 = m64[4] ^ base[4], c6 = m64[5] ^ base[5], c7 = m64[6] ^ base[6], c8 = m64[7] ^ base[7];
    for (unsigned i = 8; i < 32; ++i)
        m64[i] = gamma(m64[i - 8], m64[i - 4]) + gamma(m64[i - 7], m64[i - 3]) + gamma(m64[i - 6], m64[i - 2]) + gamma(m64[i - 5], m64[i - 1]);
    for (unsigned i = 24; i < 32; ++i)
        m128[i] = U128(m64[i]);
    for (unsigned i = 32; i < 48; ++i)
        m128[i] = beta(m128[i - 8]) + beta(m128[i - 7]) + beta(m128[i - 4]) + beta(m128[i - 3]);
    for (unsigned i = 41; i < 48; ++i)
        m256[i] = U256(m128[i]);
    for (unsigned i = 48; i < 64; ++i)
        m256[i] = beta(m256[i - 7]) + beta(m256[i - 6]) + beta(m256[i - 2]) + beta(m256[i - 1]);
    for (unsigned i = 0; i < 32; ++i) {
        auto powers = exp64.pair(c1 ^ sig0(c2), c3 ^ sig1(c4));
        auto d1 = powers.first + ror<7>(c6) + rol<13>(c5);
        auto d2 = powers.second + ch(c5, c6, c7) + maj(c1, c4, c8);
        auto d3 = c8 + Sig0(c7) + round64[i];
        auto d4 = c5 + Sig1(c1) + d1 + m64[i] + t.pow64(d3);
        c8 = c7 + U64(folded);
        c7 = c6 + d1 + sig0(M64::mul(d1, d3));
        c6 = c5 + sig1(d2 ^ d3);
        c5 = c4;
        c4 = c3 + d4;
        c3 = c2;
        c2 = c1 ^ d3;
        c1 = gamma(d1, d2) + (d3 ^ d4) + beta(M64::mul(d2, d4));
    }
    U128 f1, f2, f3, f4;
    f1.v = {c5.v[0], c1.v[0]};
    f2.v = {c6.v[0], c2.v[0]};
    f3.v = {c7.v[0], c3.v[0]};
    f4.v = {c8.v[0], c4.v[0]};
    for (unsigned i = 0; i < 16; ++i) {
        auto powers = exp128.pair(f1 ^ beta(f3), f2 ^ beta(f4));
        auto a = powers.first + sig0(f2 ^ f4) + ch(f2, f3, f4);
        auto b = powers.second + sig1(f1 ^ f3) + maj(f1, f2, f3);
        auto c = rol<29>(~f1) + ror<7>(~f2) + f3 + (~f4) + m128[32 + i] + t.pow128(f3);
        auto d = rol<29>(~f3) + ror<7>(f4) + f2 + (~f1) + round128[i];
        f4 = f3 + a;
        f3 = f2;
        f2 = f1 + c + beta(d);
        f1 = b + Sig0(a ^ c ^ d);
    }
    U256 h1, h2;
    h1.v = {f3.v[0], f3.v[1], f1.v[0], f1.v[1]};
    h2.v = {f4.v[0], f4.v[1], f2.v[0], f2.v[1]};
    for (unsigned i = 0; i < 16; ++i) {
        auto powers = exp256.pair(rol<11>(h1) ^ h2, h1 ^ beta(~h2));
        auto j1 = powers.first + beta(h1 ^ h2);
        auto j2 = powers.second + gamma(h1, h2);
        powers = exp256.pair(j1, j2);
        auto j3 = gamma(j1, j2) + powers.first + round256[i];
        auto j4 = gamma(j3, beta(h2)) + powers.second + m256[48 + i] + ch(j1, j2, j3);
        auto [j5, j6] = t.curve[0](h1);
        auto [j7, j8] = t.curve[1](h2);
        auto [j9, j10] = t.curve[2](j1);
        auto [j11, j12] = t.curve[3](j3);
        h2 = h1 + gamma(j1, j3) + maj(j2, j3, j4) + j5 + j8 + ch(j1, j6, j7) + beta(j10) + j11;
        h1 = beta(j2) + gamma(j3, j4) + Sig1(M256::mul(j1, j2)) + maj(j7, j6, j5) + Sig0(j9) + j12;
    }
    auto u1 = h1, u2 = h2 + U256(folded);
    for (const auto& fc : final) {
        auto powers = t.pow256.pair(u1, u2);
        auto k1 = powers.first + beta(u2);
        auto k2 = powers.second + gamma(u1, u2) + ch(k1, u2, u1);
        auto k3 = inverse256(k2) + beta(u1) + beta(u2 ^ k1 ^ fc);
        u2 = (k1 ^ u1) + maj(k1, k2, u2) + beta(k3);
        u1 = beta(k1) + k2;
    }
    Digest result;
    (u1 + beta(u2)).store(result.data());
    return result;
}
struct Fold {
    U512 c0 = BC3FOLD_CONST;
    U256 s1 = U256(shr<256>(BC3FOLD_CONST)), s2 = U256(BC3FOLD_CONST);
    U512 length; // byte count; serialized as a 512-bit bit count at finish
    std::array<std::uint8_t, 64> buffer{};
    std::size_t used = 0;
    void block(const void* data)
    {
        auto dt = U512::load(data);
        c0 = fold_power()(c0) + ror<237>(c0);
        auto c1 = U256(shr<256>(c0)), c2 = U256(c0);
        auto d0 = exp512(c0 ^ dt) + rol<291>(dt);
        auto f1 = U256(shr<256>(d0)) ^ s1, f2 = U256(d0) ^ s2;
        const auto& t = tables();
        const auto& g = generator_table();
        for (unsigned i = 0; i < 64; ++i) {
            const auto previous = c1;
            c1 = t.pow256(c2) + beta(c1);
            c2 = t.pow256(previous) + beta(c2);
            auto w1 = exp256(f1) + beta(f2);
            auto w2 = t.pow256(f2) + beta(f1) + c2;
            auto w3 = f1 + f2 + c1;
            auto [x1, y1] = g(w1);
            auto [x2, y2] = g(w2);
            auto [x3, y3] = g(w3);
            f2 = beta(w2) + gamma(y1, w1) + ch(y2, w1, f2) + w3;
            f1 = f2 + sig0(x2) + Sig1(y2) + w1 + maj(w2, y1, x2) + ch(y3, x1, y2) + x3;
        }
        c0.v = {c2.v[0], c2.v[1], c2.v[2], c2.v[3], c1.v[0], c1.v[1], c1.v[2], c1.v[3]};
        s1 = s1 + f1;
        s2 = s2 + f2;
    }
    void update(const void* ptr, std::size_t n)
    {
        if (!n) return;
        auto next = length + U512(n);
        if (next < length || (next.v[7] >> 61)) throw std::length_error("BC3Fold input exceeds 512-bit bit count");
        length = next;
        const auto* data = static_cast<const std::uint8_t*>(ptr);
        if (used) {
            const auto count = std::min(n, 64 - used);
            std::memcpy(buffer.data() + used, data, count);
            used += count;
            data += count;
            n -= count;
            if (used == 64) {
                block(buffer.data());
                used = 0;
            }
        }
        while (n >= 64) {
            block(data);
            data += 64;
            n -= 64;
        }
        if (n) {
            std::memcpy(buffer.data(), data, n);
            used = n;
        }
    }
    Block512 finish()
    {
        buffer[used] = 0x80;
        std::fill(buffer.begin() + std::ptrdiff_t(used) + 1, buffer.end(), 0);
        block(buffer.data());
        // Length is less than 2^509 bytes, so rotating left by three is a shift.
        rol<3>(length).store(buffer.data());
        block(buffer.data());
        Block512 result;
        s1.store(result.data());
        beta(s2).store(result.data() + 32);
        return result;
    }
};
} // namespace bc3::detail
namespace bc3 {
void configure(Mode mode)
{
    const unsigned window = mode == Mode::compact ? 10 : (mode == Mode::fast ? 16 : 0);
    if (!window) throw std::invalid_argument("Unknown BC3 mode");
    std::lock_guard<std::mutex> lock(detail::config_mutex);
    if (detail::configuration_locked && detail::configured_window.load(std::memory_order_relaxed) != window)
        throw std::logic_error("BC3 mode cannot change after initialization");
    detail::configured_window.store(window, std::memory_order_relaxed);
}
void initialize() { (void)detail::tables(); }
BuildInfo build_info() noexcept
{
    constexpr bool gmp = false;
    const unsigned w = detail::configured_window.load(std::memory_order_relaxed), d = (256 + w - 1) / w;
    const std::size_t ecbytes = 4 * 64 * ((d - 1) * ((std::size_t{1} << w) - 1) + ((std::size_t{1} << (256 - w * (d - 1))) - 1));
    return {w, gmp, sizeof(detail::Tables) + ecbytes};
}
Digest bc3mix512(const void* input)
{
    if (!input) throw std::invalid_argument("bc3mix512 requires 64 input bytes");
    return detail::mix512(input, 0);
}
Block512 bc3fold(const void* data, std::size_t n)
{
    if (n && !data) throw std::invalid_argument("Null BC3 input");
    detail::Fold fold;
    fold.update(data, n);
    return fold.finish();
}
Digest bc3mix(const void* data, std::size_t n)
{
    if (n == 64) return bc3mix512(data);
    const auto folded = bc3fold(data, n);
    return detail::mix512(folded.data(), 1);
}
struct Mix::Impl {
    std::array<std::uint8_t, 64> first{};
    std::size_t size = 0;
    std::unique_ptr<detail::Fold> fold;
    bool done = false;
};
Mix::Mix() : impl_(std::make_unique<Impl>()) {}
Mix::~Mix() = default;
Mix::Mix(Mix&&) noexcept = default;
Mix& Mix::operator=(Mix&&) noexcept = default;
void Mix::update(const void* data, std::size_t n)
{
    if (!impl_ || impl_->done) throw std::logic_error("BC3Mix state already finished or moved");
    if (n && !data) throw std::invalid_argument("Null BC3 input");
    if (!n) return;
    auto& s = *impl_;
    if (s.fold) {
        s.fold->update(data, n);
        return;
    }
    if (n <= 64 - s.size) {
        std::memcpy(s.first.data() + s.size, data, n);
        s.size += n;
        return;
    }
    s.fold = std::make_unique<detail::Fold>();
    s.fold->update(s.first.data(), s.size);
    s.fold->update(data, n);
}
Digest Mix::finish()
{
    if (!impl_ || impl_->done) throw std::logic_error("BC3Mix state already finished or moved");
    auto& s = *impl_;
    s.done = true;
    if (!s.fold) return bc3mix(s.first.data(), s.size);
    auto b = s.fold->finish();
    return detail::mix512(b.data(), 1);
}
std::string hex(const Digest& digest)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string s(64, '0');
    for (std::size_t i = 0; i < 32; ++i) {
        s[2 * i] = digits[digest[i] >> 4];
        s[2 * i + 1] = digits[digest[i] & 15];
    }
    return s;
}
} // namespace bc3
