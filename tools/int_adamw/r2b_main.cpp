// R2b C++ leg — thin driver over the ax::nn::ib engine (the guts
// moved to src/nn/intbirth.cpp for the Python module; this driver
// re-verifying the certified milestone digests certifies the
// refactor, per the booked pattern). Reference:
// llmopt scratch/detbwd_r2b.py + scratch/detbwd_r2b_ref/r2b_ref.json.
//
// Build: c++ -O2 -std=c++17 -I../../include r2b_main.cpp \
//            ../../src/nn/intbirth.cpp -o r2b
// Run:   ./r2b r2b_init.bin r2b_tables.bin
#include <ax/nn/intbirth.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

static std::string slurp(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s r2b_init.bin r2b_tables.bin\n",
                 argv[0]);
    return 1;
  }
  ax::nn::ib::contract c;  // defaults == the r2b_ref.json contract
  ax::nn::ib::full_birth fb(slurp(argv[2]), slurp(argv[1]), c);
  const int STEPS = 1000, EVERY = 125;
  long long loss0 = 0, loss_mid = 0;
  for (int step = 1; step <= STEPS; step++) {
    fb.run(1);
    if (step == 1) loss0 = fb.last_loss();
    if (step == STEPS / 2 + 1) loss_mid = fb.last_loss();
    if (step % EVERY == 0)
      std::printf("[r2b-cpp] step %d loss %lld nz %.3f traj-sha %s\n",
                  step, (long long)fb.last_loss(), fb.nz_last(),
                  fb.mark().c_str());
  }
  const long long ll = fb.last_loss();
  std::printf("[r2b-cpp] loss %lld -> %lld -> %lld  falling: %s\n",
              loss0, loss_mid, ll,
              ll < loss_mid && loss_mid < loss0 ? "true" : "false");
  std::printf("[r2b-cpp] FINAL trajectory sha %s\n",
              fb.traj_sha().c_str());
  return 0;
}
