#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wheel_dir="${WHEEL_DIR:-${script_dir}/wheels}"
mkdir -p -- "$wheel_dir"
cd -- "$wheel_dir"

while IFS= read -r url; do
    filename="${url##*/}"
    filename="${filename//%2B/+}"

    echo "正在下载：$filename"
    wget --continue --tries=10 --timeout=60 --waitretry=5 \
        --output-document="$filename" "$url"
done <<'URLS'
https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl
https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl
https://files.pythonhosted.org/packages/82/df/4b01f10069e23c641f116c62fc31e31e8dc361a153175d81561d15c8143b/nvidia_cublas_cu12-12.8.3.14-py3-none-manylinux_2_27_x86_64.whl
https://files.pythonhosted.org/packages/39/6f/3683ecf4e38931971946777d231c2df00dd5c1c4c2c914c42ad8f9f4dca6/nvidia_cuda_cupti_cu12-12.8.57-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
https://files.pythonhosted.org/packages/d4/22/32029d4583f7b19cfe75c84399cbcfd23f2aaf41c66fc8db4da460104fff/nvidia_cuda_nvrtc_cu12-12.8.61-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
https://files.pythonhosted.org/packages/16/f6/0e1ef31f4753a44084310ba1a7f0abaf977ccd810a604035abb43421c057/nvidia_cuda_runtime_cu12-12.8.57-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
https://files.pythonhosted.org/packages/25/dc/dc825c4b1c83b538e207e34f48f86063c88deaa35d46c651c7c181364ba2/nvidia_cudnn_cu12-9.7.1.26-py3-none-manylinux_2_27_x86_64.whl
https://files.pythonhosted.org/packages/ac/26/b53c493c38dccb1f1a42e1a21dc12cba2a77fbe36c652f7726d9ec4aba28/nvidia_cufft_cu12-11.3.3.41-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
https://files.pythonhosted.org/packages/e5/9c/1f3264d0a84c8a031487fb7f59780fc78fa6f1c97776233956780e3dc3ac/nvidia_cufile_cu12-1.13.0.11-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
https://files.pythonhosted.org/packages/bd/fc/7be5d0082507269bb04ac07cc614c84b78749efb96e8cf4100a8a1178e98/nvidia_curand_cu12-10.3.9.55-py3-none-manylinux_2_27_x86_64.whl
https://files.pythonhosted.org/packages/c2/08/953675873a136d96bb12f93b49ba045d1107bc94d2551c52b12fa6c7dec3/nvidia_cusolver_cu12-11.7.2.55-py3-none-manylinux_2_27_x86_64.whl
https://files.pythonhosted.org/packages/c2/ab/31e8149c66213b846c082a3b41b1365b831f41191f9f40c6ddbc8a7d550e/nvidia_cusparse_cu12-12.5.7.53-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
https://files.pythonhosted.org/packages/3b/9a/72ef35b399b0e183bc2e8f6f558036922d453c4d8237dab26c666a04244b/nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl
https://files.pythonhosted.org/packages/67/ca/f42388aed0fddd64ade7493dbba36e1f534d4e6fdbdd355c6a90030ae028/nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
https://files.pythonhosted.org/packages/03/f8/9d85593582bd99b8d7c65634d2304780aefade049b2b94d96e44084be90b/nvidia_nvjitlink_cu12-12.8.61-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
https://files.pythonhosted.org/packages/8d/cd/0e8c51b2ae3a58f054f2e7fe91b82d201abfb30167f2431e9bd92d532f42/nvidia_nvtx_cu12-12.8.55-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
https://files.pythonhosted.org/packages/24/5f/950fb373bf9c01ad4eb5a8cd5eaf32cdf9e238c02f9293557a2129b9c4ac/triton-3.3.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
URLS

echo "下载完成：$(pwd)"