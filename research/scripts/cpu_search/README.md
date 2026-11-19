# Search on CPUs: the measurements behind research/README.md

All four scripts score `zero2-dst2` step 5000, rebuilt from the app's float16
export in `packages/engine/src/rl/zero/`. Run them from this directory, with
`research/py` on the path and one thread per process:

```bash
export PYTHONPATH=$PWD/../../py:$PWD OMP_NUM_THREADS=1
python load_zero.py                          # the rebuild loads; lists missing tensors
python bench_search.py 60 12 4 ref,fast,bf16 # ms per searched decision ("Search on CPUs")
python try_onnx.py                           # ONNX Runtime against torch (needs onnxruntime, onnxscript)
python label_noise.py 32 4 out.json 2000     # search's label self-agreement, 8x3 against 12x4
```

`label_noise.py HANDS WORKERS OUT SEED`: two runs pool when their seeds differ.
