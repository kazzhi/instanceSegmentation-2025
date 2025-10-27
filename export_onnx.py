import argparse
from pathlib import Path
from typing import Dict, List

import torch
import onnx
import onnxruntime as ort

from data import cfg, set_cfg, set_dataset, mask_type
from utils.functions import SavePath
from yolact import Yolact
from yolact import YolactExportWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export Yolact heads to ONNX for TensorRT pipelines.')
    parser.add_argument('--trained_model', required=True,
                        help='Path to the pretrained Yolact checkpoint (.pth).')
    parser.add_argument('--config', default=None,
                        help='Config name to load (e.g., yolact_resnet50_config).')
    parser.add_argument('--dataset', default=None,
                        help='Dataset config override (e.g., coco2017_dataset).')
    parser.add_argument('--output', default='yolact_pred.onnx',
                        help='Destination ONNX file path.')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version to target.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Dummy batch size used for export.')
    parser.add_argument('--height', type=int, default=550,
                        help='Input tensor height.')
    parser.add_argument('--width', type=int, default=550,
                        help='Input tensor width.')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Export using CUDA tensors if available.')
    parser.add_argument('--cpu', dest='cuda', action='store_false',
                        help='Force CPU export.')
    parser.set_defaults(cuda=torch.cuda.is_available())
    parser.add_argument('--skip_ort', action='store_true',
                        help='Skip ONNX Runtime validation step.')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Absolute tolerance for ONNX runtime output checks.')
    return parser.parse_args()


def prepare_configuration(args: argparse.Namespace) -> None:
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        inferred = f'{model_path.model_name}_config'
        print(f'Config not specified; inferring {inferred} from checkpoint name.')
        set_cfg(inferred)
    else:
        set_cfg(args.config)

    if args.dataset is not None:
        set_dataset(args.dataset)


def build_model(args: argparse.Namespace, device: torch.device) -> Yolact:
    net = Yolact()
    state_dict = torch.load(args.trained_model, map_location=device)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    net.freeze_bn(enable=False)
    net = net.to(device)
    return net


def determine_outputs() -> List[str]:
    names = ['loc', 'conf', 'mask', 'priors']
    if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        names.append('proto')
    if cfg.use_mask_scoring:
        names.append('score')
    if cfg.use_instance_coeff:
        names.append('inst')
    return names


def dynamic_axes_for(outputs: List[str]) -> Dict[str, Dict[int, str]]:
    axes = {'images': {0: 'batch'}}
    for name in outputs:
        if name in {'priors'}:
            continue
        axes[name] = {0: 'batch'}
    return axes


def run_export(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    if args.cuda and device.type != 'cuda':
        raise RuntimeError('CUDA requested but not available.')

    prepare_configuration(args)
    net = build_model(args, device)
    wrapper = YolactExportWrapper(net).to(device)
    wrapper.eval()

    dummy = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    output_names = determine_outputs()
    dynamic_axes = dynamic_axes_for(output_names)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            input_names=['images'],
            output_names=output_names,
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True
        )

    print(f'ONNX graph written to {args.output}')

    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    print('onnx.checker.check_model: OK')

    if args.skip_ort:
        return

    wrapper_cpu = YolactExportWrapper(build_model(args, torch.device('cpu')))
    wrapper_cpu.eval()
    dummy_cpu = torch.randn(args.batch_size, 3, args.height, args.width)

    with torch.no_grad():
        torch_outputs = wrapper_cpu(dummy_cpu)
        torch_outputs = [t.detach().cpu().numpy() for t in torch_outputs]

    ort_session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    ort_outputs = ort_session.run(None, {'images': dummy_cpu.numpy()})

    for name, torch_out, ort_out in zip(output_names, torch_outputs, ort_outputs):
        if not torch.allclose(torch.tensor(ort_out), torch.tensor(torch_out), atol=args.tolerance, rtol=1e-3):
            diff = (torch.tensor(ort_out) - torch.tensor(torch_out)).abs().max().item()
            raise RuntimeError(f'ONNXRuntime output mismatch for {name}: max diff {diff:.3e}')

    print('ONNXRuntime validation: OK')


def main() -> None:
    args = parse_args()
    run_export(args)


if __name__ == '__main__':
    main()
