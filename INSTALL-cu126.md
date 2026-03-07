# Installtion for cu126

In nunif, PyTorch `cu128` (CUDA 12.8) is used by default because it is the only version that supports the RTX 50xx series.

However, `cu128` has dropped support for GPUs from the Volta (Tesla V100, Titan V), Pascal (GTX 10xx, Tesla P40, Tesla P4), and Maxwell (GeForce 9xx, Titan X) generations.
`cu126` still supports those GPUs.

If you want to use `cu126`, use `requirements-torch-cu126.txt` instead of `requirements-torch.txt`.

```
pip3 install -r requirements-torch-cu126.txt
```

| PyTorch CUDA Version | Compute Capability  |
|----------------------|---------------------|
| `cu128`              | `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120`, `compute_120`|
| `cu126`              | `sm_50`, `sm_60`, `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_90`|
