"""Exercise GPU kernels used by training; CUDA availability alone is insufficient."""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', help='Optional trusted YOLO checkpoint to test on GPU')
    args = p.parse_args()
    import torch
    import torchvision
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    device = torch.device('cuda:0')
    assert torch.version.cuda == '12.8', torch.version.cuda
    layer = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.001)
    x = torch.randn(2, 3, 64, 64, device=device)
    loss = layer(x).square().mean()
    loss.backward()
    optimizer.step()
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    torch.cuda.synchronize()
    assert keep.cpu().tolist() == [0, 2]
    if args.weights:
        import numpy as np
        from ultralytics import YOLO
        model = YOLO(args.weights)
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=0, verbose=False)
        torch.cuda.synchronize()
    print(json.dumps({'status': 'PASS', 'gpu': torch.cuda.get_device_name(0),
                      'compute_capability': torch.cuda.get_device_capability(0),
                      'torch': torch.__version__, 'torchvision': torchvision.__version__,
                      'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                      'checks': ['conv_forward', 'backward', 'optimizer', 'cuda_nms']}, indent=2))


if __name__ == '__main__':
    main()
