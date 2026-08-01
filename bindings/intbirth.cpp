/** pybind11 module `intbirth` — the integer-birth engine exposed to
    house Python (relay 2026-08-01 engine ask). Protocol unchanged:
    tables/init travel as opaque bytes, the contract dict carries the
    r2b_ref.json constants (+ dims), milestone digests are computed
    engine-side (mark()) and compared house-side. run() releases the
    GIL — the whole engine is int64 + shipped tables, no Python
    callbacks.

      fb = intbirth.FullBirth(tables_bytes, init_bytes, contract)
      fb.run(125); fb.mark()   # -> hex digest at the milestone
      fb.loss, fb.nz, fb.step_count
      fb.weights_bytes()       # wide Q_w weights, KEYS order, i64 LE
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ax/nn/intbirth.hpp>

#include <map>
#include <string>

namespace py = pybind11;
using ax::nn::ib::contract;
using ax::nn::ib::full_birth;

namespace {

contract contract_from_dict(const py::dict& d) {
  contract c;
  const auto geti = [&](const char* k, int dflt) {
    return d.contains(k) ? d[k].cast<int>() : dflt;
  };
  const auto getl = [&](const char* k, long long dflt) {
    return d.contains(k) ? d[k].cast<long long>() : dflt;
  };
  c.T = geti("T", c.T);
  c.D = geti("D", c.D);
  c.DH = geti("DH", c.DH);
  c.F = geti("F", c.F);
  c.V = geti("V", c.V);
  c.shift = geti("SHIFT", c.shift);
  c.gboost = getl("GBOOST", c.gboost);
  c.pq = getl("PQ", c.pq);
  c.act_clamp = getl("ACT_CLAMP", c.act_clamp);
  c.eps32 = getl("EPS32", c.eps32);
  c.lrn = getl("LRN", c.lrn);
  c.lrd = getl("LRD", c.lrd);
  return c;
}

}  // namespace

PYBIND11_MODULE(intbirth, m) {
  m.doc() = "axiom integer-birth engine (R2b full-block, contract-"
            "parameterized; digests engine-side, comparison house-side)";
  py::class_<full_birth>(m, "FullBirth")
      .def(py::init([](const py::bytes& tables, const py::bytes& init,
                       const py::dict& c) {
             return new full_birth(std::string(tables),
                                   std::string(init),
                                   contract_from_dict(c));
           }),
           py::arg("tables_bytes"), py::arg("init_bytes"),
           py::arg("contract"))
      .def("run", &full_birth::run, py::arg("steps"),
           py::call_guard<py::gil_scoped_release>())
      .def("mark", &full_birth::mark)
      .def("traj_sha", &full_birth::traj_sha)
      .def("milestone_sha", &full_birth::traj_sha,
           "alias of traj_sha (the relay-proposed name)")
      .def("weights_bytes",
           [](const full_birth& fb) {
             return py::bytes(fb.weights_bytes());
           })
      .def_property_readonly("step_count", &full_birth::step_count)
      .def_property_readonly("loss", &full_birth::last_loss)
      .def_property_readonly("nz", &full_birth::nz_last);
}
