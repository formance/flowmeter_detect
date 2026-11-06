FROM flowmeter_infer:v1.0

ARG PIP_INDEX_URL=https://pypi.org/simple

RUN python -m pip install --no-cache-dir --only-binary=:all: --timeout 20 --retries 2 --index-url "${PIP_INDEX_URL}" \
    onnx==1.17.0 onnxruntime==1.22.1 \
    && python -m pip check \
    && python -c "import cv2, numpy, onnx, onnxruntime as ort; assert 'CPUExecutionProvider' in ort.get_available_providers(); print('ONNX:', onnx.__version__, 'ONNX Runtime:', ort.__version__, 'NumPy:', numpy.__version__, 'OpenCV:', cv2.__version__)"
