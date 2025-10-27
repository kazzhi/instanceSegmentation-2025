
````markdown
# YOLACT TensorRT Quantization and FP16 Conversion

This repository provides instructions for setting up the environment and building the TensorRT engine for YOLACT using FP16 precision.

---

## Environment Setup

Set up the Conda environment:

```bash
conda env create -f environment.yml -n yolact-quant
conda activate yolact-quant
````

---

## Building the TensorRT Engine Builder

Compile the `trt_force_fp16.cpp` file:

```bash
g++ -std=c++17 trt_force_fp16.cpp \
  -I${TENSORRT_ROOT}/include -I${CUDA_ROOT}/include \
  -L${TENSORRT_ROOT} -L${CUDA_ROOT}/lib64 \
  -lnvinfer -lnvonnxparser -lcudart \
  -o trt_force_fp16
```

If you are using GCC ≤ 8, add the following flag:

```bash
-lstdc++fs
```

If you later encounter unresolved plugin symbols, add:

```bash
-lnvinfer_plugin
```

---

## Troubleshooting Header and Library Recognition

If header files or libraries are not recognized, set the library path manually:

```bash
export LD_LIBRARY_PATH=${TENSORRT_ROOT}:${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH
```

---

## Running the Converter

Run the FP16 conversion:

```bash
export LD_LIBRARY_PATH=${TENSORRT_ROOT}:${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH

./trt_force_fp16 \
  --onnx yolact_pred.onnx \
  --engine yolact_int8.plan \
  --fp16-tensor proto
```

This converts the ONNX model `yolact_pred.onnx` into a TensorRT int8 engine `yolact_fp16.plan` while enforcing FP16 precision on the specified tensor `proto`.

---

## Notes

* Ensure CUDA and TensorRT versions are compatible with your installed GPU driver.
* If you encounter `CUDA initialization failure (error: 35)`, verify that:

  * The NVIDIA driver is installed correctly.
  * `libcuda.so` is loaded from `/usr/lib/x86_64-linux-gnu/` rather than the CUDA `compat` directory.

```
```
