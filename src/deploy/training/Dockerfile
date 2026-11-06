ARG BASE_IMAGE=m.daocloud.io/docker.io/library/ubuntu:24.04
FROM ${BASE_IMAGE} AS python-builder
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12.12
COPY install-missing.sh /tmp/install-missing.sh
RUN sh /tmp/install-missing.sh ca-certificates xz-utils build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev
COPY wheels_python/Python-${PYTHON_VERSION}.tar.xz /tmp/python.tar.xz
COPY wheels_python/Python-${PYTHON_VERSION}.tar.xz.sha256 /tmp/python.tar.xz.sha256
RUN cd /tmp && sha256sum -c python.tar.xz.sha256 \
    && tar -xJf python.tar.xz \
    && cd /tmp/Python-${PYTHON_VERSION} \
    && ./configure --prefix=/opt/python --with-ensurepip=install \
    && make -j2 && make install \
    && rm -rf /tmp/Python-* /tmp/python.tar.xz /tmp/python.tar.xz.sha256

FROM ${BASE_IMAGE} AS runtime-base
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/python/bin:$PATH MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics YOLO_AUTOINSTALL=false \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY install-missing.sh /tmp/install-missing.sh
RUN sh /tmp/install-missing.sh ca-certificates libssl3t64 zlib1g libbz2-1.0 \
    libreadline8t64 libsqlite3-0 libffi8 liblzma5 libgomp1 libstdc++6 libgl1 libglib2.0-0t64 \
    && rm /tmp/install-missing.sh
WORKDIR /app

FROM runtime-base AS python-deps
ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY --from=python-builder /opt/python /opt/python
COPY requirements.txt offline-wheels.json check_offline_wheels.py ./
COPY wheels/*.whl /tmp/wheels/
# All CUDA/Triton wheels must exist locally and match official hashes.
# CUDA runtime libraries come from these wheels; no CUDA base image is required.
# Small application dependencies can still come from the configured mirror.
RUN python3.12 check_offline_wheels.py --manifest offline-wheels.json \
    --directory /tmp/wheels --output /tmp/offline-requirements.txt \
    && python3.12 -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
    --find-links=/tmp/wheels -r /tmp/offline-requirements.txt -r requirements.txt \
    && python3.12 -m pip check \
    && python3.12 -c "import sys, ssl, torch, torchvision, ultralytics; assert sys.version_info[:3] == (3, 12, 12); assert torch.__version__ == '2.7.1+cu128'; assert torchvision.__version__ == '0.22.1+cu128'; assert torch.version.cuda == '12.8'; print(sys.version, ssl.OPENSSL_VERSION, torch.__version__, torchvision.__version__, ultralytics.__version__)" \
    && python3.12 -c "import onnx, onnxruntime; print('ONNX:', onnx.__version__, 'ONNX Runtime:', onnxruntime.__version__)"

FROM runtime-base
COPY --from=python-deps /opt/python /opt/python
# Mount training code and datasets when starting the container.
CMD ["bash"]
