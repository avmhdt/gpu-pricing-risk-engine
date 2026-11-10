// Philox4x32 known-answer tests against the shared fixture (FR-02, AC-W-06): the host C++
// build of philox4x32.cuh must reproduce the Random123 vectors bit for bit.
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "gpre/baseline.hpp"
#include "philox4x32.cuh"

namespace {

std::uint32_t hex_word(const std::string& s) {
  return static_cast<std::uint32_t>(std::stoul(s, nullptr, 16));
}

nlohmann::json load_kat() {
  std::ifstream in(GPRE_KAT_PATH);
  EXPECT_TRUE(in.good()) << "cannot open " << GPRE_KAT_PATH;
  return nlohmann::json::parse(in);
}

}  // namespace

TEST(Philox, KnownAnswerVectors) {
  const auto kat = load_kat();
  ASSERT_EQ(kat["vectors"].size(), 6U);
  for (const auto& v : kat["vectors"]) {
    gpre::Philox4x32Counter ctr{};
    gpre::Philox4x32Key key{};
    for (int i = 0; i < 4; ++i) {
      ctr.v[i] = hex_word(v["counter"][static_cast<std::size_t>(i)].get<std::string>());
    }
    for (int i = 0; i < 2; ++i) {
      key.v[i] = hex_word(v["key"][static_cast<std::size_t>(i)].get<std::string>());
    }
    const gpre::Philox4x32Counter out = gpre::philox4x32(ctr, key, v["rounds"].get<int>());
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(out.v[i], hex_word(v["expected"][static_cast<std::size_t>(i)].get<std::string>()))
          << "rounds " << v["rounds"] << " word " << i;
    }
  }
}

TEST(Philox, SeedSplitsIntoLowAndHighWords) {
  const gpre::Philox4x32Key key = gpre::philox_key_from_seed(0x0123456789ABCDEFULL);
  EXPECT_EQ(key.v[0], 0x89ABCDEFU);
  EXPECT_EQ(key.v[1], 0x01234567U);
  const auto words = gpre::philox_words(0, {0, 0, 0, 0});
  EXPECT_EQ(words[0], 0x6627e8d5U);
}

// Counter-based output must not depend on which thread computes it.
TEST(Philox, ThreadCountInvariance) {
  constexpr std::size_t kN = std::size_t{1} << 14U;
  auto fill = [](std::vector<std::uint32_t>& out, std::size_t first, std::size_t stride) {
    for (std::size_t p = first; p < kN; p += stride) {
      const auto w = gpre::philox_words(20261030ULL, {0, static_cast<std::uint32_t>(p), 7, 0});
      for (std::size_t i = 0; i < 4; ++i) {
        out[(p * 4) + i] = w[i];
      }
    }
  };
  std::vector<std::uint32_t> serial(kN * 4);
  fill(serial, 0, 1);
  for (std::size_t n_threads : {std::size_t{1}, std::size_t{4}, std::size_t{8}}) {
    std::vector<std::uint32_t> parallel(kN * 4);
    {
      std::vector<std::jthread> pool;
      for (std::size_t t = 0; t < n_threads; ++t) {
        pool.emplace_back([&parallel, &fill, t, n_threads] { fill(parallel, t, n_threads); });
      }
    }
    EXPECT_EQ(parallel, serial) << n_threads << " threads";
  }
}
