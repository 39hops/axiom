#pragma once
/** @file fft.hpp Iterative radix-2 FFT (complex) and NTT (mod p). */
#include <complex>
#include <cstdint>
#include <vector>

namespace ax {

/** Default NTT prime 998244353 = 119 * 2^23 + 1, primitive root 3. */
inline constexpr std::uint64_t ntt_mod = 998244353;
/** Second NTT prime 167772161 = 5 * 2^25 + 1, primitive root 3. */
inline constexpr std::uint64_t ntt_mod2 = 167772161;

/** In-place iterative radix-2 FFT. Size must be a power of two.
    @param invert true for inverse transform (includes 1/n scaling). */
void fft(std::vector<std::complex<double>>& a, bool invert);

/** Real convolution via FFT (floating point; rounding not exact). */
std::vector<double> convolve(const std::vector<double>& a,
                             const std::vector<double>& b);

/** Exact convolution mod prime (must be k*2^m+1 with primitive root 3 and
    transform size <= 2^m). Inputs reduced mod prime. */
std::vector<std::uint64_t> ntt_convolve(const std::vector<std::uint64_t>& a,
                                        const std::vector<std::uint64_t>& b,
                                        std::uint64_t mod = ntt_mod);

}  // namespace ax
