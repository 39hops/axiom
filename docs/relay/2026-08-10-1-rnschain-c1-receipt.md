# RNSCHAIN C1 receipt (axiom CPU leg, 2026-08-10, post anchor-v2 rebase)

Run: g++ -std=c++23 -O2, linux/gcc 15.2 LP64 worker (3080 host, CPU only).
RNS arm rides the surviving anchor-v2 ring core (ax::rns ctx/addm/mulm);
centered CRT exit is oracle-local via ax::crt. Digests identical to the
pre-rebase run - port is semantics-preserving.
All classes x depths PASS; perm bit-identity depths 1 and 8 PASS.

```jsonl
{"receipt":"rnschain-c1","n":32,"channels":8,"seed":"0x20260810524e53","input_contract":"_f24: all entries are integers in [-2^23, 2^23-1], exactly representable in an fp32/fp64 significand","perm_bitidentity":[1,8],"primes":[2305843009213693951,2305843009213693921,2305843009213693907,2305843009213693723,2305843009213693693,2305843009213693669,2305843009213693613,2305843009213693561]}
{"class":"random","depth":2,"pass":true,"digest":"7e115e45f0583fe5","cum_ms":65.9}
{"class":"random","depth":4,"pass":true,"digest":"a1f93ac203cec5dd","cum_ms":136.5}
{"class":"random","depth":6,"pass":true,"digest":"49154a9d2415456b","cum_ms":214.0}
{"class":"random","depth":8,"pass":true,"digest":"40a03689315d4bcc","cum_ms":302.2}
{"class":"random","depth":12,"pass":true,"digest":"2a7e6dccc19a4cc1","cum_ms":411.8}
{"class":"all_max","depth":2,"pass":true,"digest":"e3c40d9b89816965","cum_ms":65.8}
{"class":"all_max","depth":4,"pass":true,"digest":"d84b34cb049b14a5","cum_ms":135.9}
{"class":"all_max","depth":6,"pass":true,"digest":"5d4678d98fe72e45","cum_ms":212.9}
{"class":"all_max","depth":8,"pass":true,"digest":"1710cf9ae2a7c465","cum_ms":300.5}
{"class":"all_max","depth":12,"pass":true,"digest":"5a58f901d93a30e5","cum_ms":410.0}
{"class":"all_min","depth":2,"pass":true,"digest":"05ea837d921da425","cum_ms":66.2}
{"class":"all_min","depth":4,"pass":true,"digest":"19e9dc490cb69ca5","cum_ms":136.3}
{"class":"all_min","depth":6,"pass":true,"digest":"ee2273ce560f3e45","cum_ms":213.9}
{"class":"all_min","depth":8,"pass":true,"digest":"06556bdc65745b25","cum_ms":301.5}
{"class":"all_min","depth":12,"pass":true,"digest":"58a41280ae7c45a5","cum_ms":410.9}
{"class":"alt_sign_extreme","depth":2,"pass":true,"digest":"04aee61318809c65","cum_ms":65.9}
{"class":"alt_sign_extreme","depth":4,"pass":true,"digest":"1c508122ac375465","cum_ms":136.4}
{"class":"alt_sign_extreme","depth":6,"pass":true,"digest":"f81656113b36b325","cum_ms":213.4}
{"class":"alt_sign_extreme","depth":8,"pass":true,"digest":"72c807b0b988bf45","cum_ms":301.1}
{"class":"alt_sign_extreme","depth":12,"pass":true,"digest":"daddde319983d9e5","cum_ms":410.3}
{"class":"sparse_zero","depth":2,"pass":true,"digest":"7b088b3ca26414f7","cum_ms":65.4}
{"class":"sparse_zero","depth":4,"pass":true,"digest":"63a4d4ed569293ee","cum_ms":136.0}
{"class":"sparse_zero","depth":6,"pass":true,"digest":"1f0416706fe6c8c4","cum_ms":213.6}
{"class":"sparse_zero","depth":8,"pass":true,"digest":"6aaae501844277c7","cum_ms":301.3}
{"class":"sparse_zero","depth":12,"pass":true,"digest":"411e35d0f4ab465e","cum_ms":410.2}
{"class":"identity_perm_stress","depth":2,"pass":true,"digest":"27d9e1e3462bc6f5","cum_ms":64.8}
{"class":"identity_perm_stress","depth":4,"pass":true,"digest":"1f8c62cfe7515bae","cum_ms":133.8}
{"class":"identity_perm_stress","depth":6,"pass":true,"digest":"9a456e8b3585508e","cum_ms":208.3}
{"class":"identity_perm_stress","depth":8,"pass":true,"digest":"0e0f6d4c08c3dd69","cum_ms":290.8}
{"class":"identity_perm_stress","depth":12,"pass":true,"digest":"884903f173e2d625","cum_ms":393.8}
{"receipt":"rnschain-c1","verdict":"PASS"}
```

shas:
```
baf60edbcaf524d8a11813d9dcec4e81c83a91df9f88b80c62974f021749272a  cuda/rns_chain.cu
2a1c522cff1d08febec2d541f48b939ad6bc0c0569e2473160ff6c6f717a162a  tools/rnschain/c1_chain_oracle.cpp
557074685077392e02456ee588865653e9e42942bd92053ca904fddfe7c3c3ca  include/ax/core/rns.hpp
```
