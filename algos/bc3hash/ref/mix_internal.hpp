#pragma once
#include "uint.hpp"

#include <utility>
namespace bc3::detail {
U256 inverse256(U256 x);
std::pair<U256, U256> curve_multiply(unsigned point, U256 scalar);
} // namespace bc3::detail
