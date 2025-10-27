import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
NETWORK_FLAGS = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build TensorRT engines (FP16 + INT8) for Yolact.')
    parser.add_argument('--onnx', default='yolact_pred.onnx',
                        help='Path to the exported Yolact ONNX graph.')
    parser.add_argument('--fp16-engine', default='yolact_fp16.plan',
                        help='Destination path for the FP16 TensorRT engine.')
    parser.add_argument('--int8-engine', default='yolact_int8.plan',
                        help='Destination path for the INT8/FP16 TensorRT engine.')
    parser.add_argument('--calib-dir', required=True,
                        help='Directory containing cached calibration tensors (.pt).')
    parser.add_argument('--calib-cache', default='yolact_int8.cache',
                        help='File used to store/load TensorRT INT8 calibration cache.')
    parser.add_argument('--max-batch', type=int, default=4,
                        help='Maximum batch size supported by the TensorRT engines.')
    parser.add_argument('--workspace-gb', type=float, default=4.0,
                        help='Workspace memory budget for TensorRT builder (in GiB).')
    parser.add_argument('--range-samples', type=int, default=32,
                        help='Number of calibration tensors to scan when estimating dynamic ranges (0 = all).')
    parser.add_argument('--skip-fp16', action='store_true',
                        help='Skip building the pure FP16 engine.')
    parser.add_argument('--skip-range-scan', action='store_true',
                        help='Skip explicit dynamic range estimation (not recommended).')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose TensorRT logging.')
    parser.add_argument('--validate', action='store_true',
                        help='Run a single ONNX vs TensorRT comparison using the first calibration tensor.')
    return parser.parse_args()


def load_calibration_files(calib_dir: Path) -> List[Path]:
    files = sorted(calib_dir.glob('*.pt'))
    if not files:
        raise FileNotFoundError(f'No .pt tensors found under {calib_dir}')
    return files


def tensor_from_file(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location='cpu')
    if not torch.is_tensor(tensor):
        raise TypeError(f'Calibration file {path} did not contain a torch.Tensor')
    return tensor


def compute_sym_range(values: Iterable[float]) -> Tuple[float, float]:
    min_val = min(values)
    max_val = max(values)
    bound = max(abs(min_val), abs(max_val))
    return -bound, bound


def estimate_dynamic_ranges(onnx_path: Path, sample_paths: Sequence[Path]) -> Tuple[Tuple[float, float], Dict[str, Tuple[float, float]]]:
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    outputs = session.get_outputs()
    ranges: Dict[str, List[float]] = {out.name: [float('inf'), float('-inf')] for out in outputs}
    input_min, input_max = float('inf'), float('-inf')

    for sample_path in sample_paths:
        tensor = tensor_from_file(sample_path)
        array = tensor.detach().cpu()
        if array.dtype == torch.float16:
            array = array.float()
        numpy_input = array.numpy()

        input_min = min(input_min, float(numpy_input.min()))
        input_max = max(input_max, float(numpy_input.max()))

        ort_outputs = session.run(None, {'images': numpy_input})
        for output_info, data in zip(outputs, ort_outputs):
            ranges[output_info.name][0] = min(ranges[output_info.name][0], float(data.min()))
            ranges[output_info.name][1] = max(ranges[output_info.name][1], float(data.max()))

    input_range = compute_sym_range([input_min, input_max])
    output_ranges = {
        name: compute_sym_range(pair)
        for name, pair in ranges.items()
    }
    return input_range, output_ranges


class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, tensors: Sequence[Path], cache_file: Path):
        super().__init__()
        self.tensors = list(tensors)
        self.cache_file = cache_file
        self.index = 0

        first = tensor_from_file(self.tensors[0]).detach().cpu()
        if first.dtype == torch.float16:
            first = first.float()
        self.shape = tuple(first.shape)
        self.host_buffer = np.ascontiguousarray(first.numpy(), dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.host_buffer.nbytes)
        cuda.memcpy_htod(self.device_input, self.host_buffer)

    def get_batch_size(self) -> int:
        return self.shape[0]

    def get_batch(self, names: Sequence[str]) -> List[int]:
        if self.index >= len(self.tensors):
            return None

        tensor = tensor_from_file(self.tensors[self.index]).detach().cpu()
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        if tuple(tensor.shape) != self.shape:
            raise ValueError(f'Calibration tensor {self.tensors[self.index]} has mismatched shape {tuple(tensor.shape)} expected {self.shape}')

        np.copyto(self.host_buffer, np.ascontiguousarray(tensor.numpy(), dtype=np.float32))
        cuda.memcpy_htod(self.device_input, self.host_buffer)

        self.index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes:
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return b''

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)


def apply_dynamic_ranges(network: trt.INetworkDefinition,
                         input_range: Optional[Tuple[float, float]],
                         output_ranges: Dict[str, Tuple[float, float]]) -> None:
    if input_range is not None:
        tensor = network.get_input(0)
        tensor.dynamic_range = input_range

    if not output_ranges:
        return

    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        if tensor.name in output_ranges:
            tensor.dynamic_range = output_ranges[tensor.name]


def build_engine(onnx_path: Path,
                 builder: trt.Builder,
                 config_callback,
                 profile_shapes: Tuple[Tuple[int, int, int, int],
                                       Tuple[int, int, int, int],
                                       Tuple[int, int, int, int]],
                 input_range: Optional[Tuple[float, float]],
                 output_ranges: Dict[str, Tuple[float, float]],
                 workspace_bytes: int) -> bytes:
    network = builder.create_network(NETWORK_FLAGS)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with onnx_path.open('rb') as f:
        if not parser.parse(f.read()):
            errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f'Failed to parse ONNX:\n{errors}')

    apply_dynamic_ranges(network, input_range, output_ranges)

    min_shape, opt_shape, max_shape = profile_shapes
    profile = builder.create_optimization_profile()
    profile.set_shape('images', min_shape, opt_shape, max_shape)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_bytes
    )
    config.add_optimization_profile(profile)
    config_callback(config)
    

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT failed to build the engine.')
    return engine


def serialize_engine(plan_bytes: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(plan_bytes)
    print(f'Wrote TensorRT engine: {destination}')


def validate_engine(engine_path: Path, onnx_path: Path, sample_path: Path) -> None:
    runtime = trt.Runtime(TRT_LOGGER)
    with engine_path.open('rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f'Failed to deserialize engine {engine_path}')

    context = engine.create_execution_context()
    bindings = []
    host_buffers = []

    sample = tensor_from_file(sample_path).detach().cpu()
    if sample.dtype == torch.float16:
        sample = sample.float()
    np_input = np.ascontiguousarray(sample.numpy(), dtype=np.float32)

    input_index = engine.get_binding_index('images')
    context.set_binding_shape(input_index, sample.shape)
    d_input = cuda.mem_alloc(np_input.nbytes)
    cuda.memcpy_htod(d_input, np_input)
    bindings.append(int(d_input))
    host_buffers.extend([np_input])

    output_bindings = []
    ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    ort_outputs = ort_session.run(None, {'images': np_input})
    ort_map = {info.name: data for info, data in zip(ort_session.get_outputs(), ort_outputs)}

    for idx in range(engine.num_bindings):
        if engine.binding_is_input(idx):
            continue
        shape = context.get_binding_shape(idx)
        dtype = trt.nptype(engine.get_binding_dtype(idx))
        size = int(np.prod(shape))
        host_output = np.empty(size, dtype=dtype)
        device_output = cuda.mem_alloc(host_output.nbytes)
        bindings.append(int(device_output))
        output_bindings.append((device_output, host_output, engine.get_binding_name(idx)))

    context.execute_v2(bindings)

    for device_buffer, host_buffer, name in output_bindings:
        cuda.memcpy_dtoh(host_buffer, device_buffer)
        engine_result = host_buffer.reshape(context.get_binding_shape(engine.get_binding_index(name)))
        ort_result = ort_map[name]
        diff = np.max(np.abs(engine_result.astype(np.float32) - ort_result.astype(np.float32)))
        print(f'[{engine_path.name}] {name}: max|Δ| = {diff:.6f}')


def main() -> None:
    args = parse_args()
    if args.verbose:
        TRT_LOGGER.min_severity = trt.Logger.VERBOSE

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f'ONNX model not found: {onnx_path}')

    calib_dir = Path(args.calib_dir).resolve()
    calibration_files = load_calibration_files(calib_dir)

    sample_tensor = tensor_from_file(calibration_files[0]).detach().cpu()
    if sample_tensor.dtype == torch.float16:
        sample_tensor = sample_tensor.float()
    batch, channels, height, width = sample_tensor.shape
    max_batch = max(batch, args.max_batch)

    profile_shapes = (
        (1, channels, height, width),
        (batch, channels, height, width),
        (max_batch, channels, height, width),
    )

    if args.skip_range_scan:
        input_range = None
        output_ranges = {}
    else:
        sample_count = len(calibration_files) if args.range_samples == 0 else min(args.range_samples, len(calibration_files))
        subset = calibration_files[:sample_count]
        print(f'Estimating dynamic ranges from {len(subset)} calibration tensors...')
        input_range, output_ranges = estimate_dynamic_ranges(onnx_path, subset)
        print(f'Input dynamic range: {input_range}')
        for name, dr in output_ranges.items():
            print(f'Output {name} dynamic range: {dr}')

    workspace_bytes = int(args.workspace_gb * (1 << 30))
    builder = trt.Builder(TRT_LOGGER)

    if not args.skip_fp16:
        print('Building FP16 engine...')

        def fp16_config(cfg: trt.IBuilderConfig) -> None:
            if not builder.platform_has_fast_fp16:
                print('Warning: Platform does not report fast FP16 support.')
            cfg.set_flag(trt.BuilderFlag.FP16)

        fp16_plan = build_engine(onnx_path, builder, fp16_config, profile_shapes, input_range, output_ranges, workspace_bytes)
        serialize_engine(fp16_plan, Path(args.fp16_engine).resolve())
    else:
        print('Skipping FP16 engine build.')

    print('Building INT8 (mixed-precision) engine...')
    calibrator = TorchEntropyCalibrator(calibration_files, Path(args.calib_cache).resolve())

    def int8_config(cfg: trt.IBuilderConfig) -> None:
        if not builder.platform_has_fast_int8:
            print('Warning: Platform does not report fast INT8 support.')
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calibrator

    int8_plan = build_engine(onnx_path, builder, int8_config, profile_shapes, input_range, output_ranges, workspace_bytes)
    serialize_engine(int8_plan, Path(args.int8_engine).resolve())

    if args.validate:
        print('Validating engines against ONNX output...')
        validate_engine(Path(args.fp16_engine), onnx_path, calibration_files[0])
        validate_engine(Path(args.int8_engine), onnx_path, calibration_files[0])


if __name__ == '__main__':
    main()
