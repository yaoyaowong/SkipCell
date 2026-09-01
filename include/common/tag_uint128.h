#ifndef COMMON_TAG_UINT128
#define COMMON_TAG_UINT128

#include <cstdint>
#include <functional>

namespace powerlaw_ann {
#pragma pack(push, 1)

struct tag_uint128_t {
  std::uint64_t data1_ = 0;
  std::uint64_t data2_ = 0;

  bool operator==(const tag_uint128_t& other) const {
    return data1_ == other.data1_ && data2_ == other.data2_;
  }

  bool operator==(std::uint64_t other) const { return data1_ == other && data2_ == 0; }

  tag_uint128_t& operator=(std::uint64_t other) {
    data1_ = other;
    data2_ = 0;

    return *this;
  }
};

#pragma pack(pop)
} // namespace powerlaw_ann

namespace std {
// Hash 128 input bits down to 64 bits of output.
// This is intended to be a reasonably good hash function.
inline std::uint64_t Hash128to64(const std::uint64_t& low, const std::uint64_t& high) {
  // Murmur-inspired hashing.
  const std::uint64_t k_mul = 0x9ddfea08eb382d69ULL;
  std::uint64_t a = (low ^ high) * k_mul;
  a ^= (a >> 47);
  std::uint64_t b = (high ^ a) * k_mul;
  b ^= (b >> 47);
  b *= k_mul;
  return b;
}

template <>
struct hash<powerlaw_ann::tag_uint128_t> {
  size_t operator()(const powerlaw_ann::tag_uint128_t& key) const noexcept {
    return Hash128to64(key.data1_, key.data2_); // map -0 to 0
  }
};

} // namespace std

#endif // COMMON_TAG_UINT128
