# KT2028 server environment

Activate: source /root/autodl-tmp/KT2028/environment/activate.sh

The Python 3.12 venv uses --system-site-packages and reuses the original miniconda torch/CUDA. Do not replace this base environment. requirements.lock includes inherited packages; it is an audit snapshot, not a portable CUDA installer. requirements.in plus constraints.txt describes the project overlay.

Default PyPI: https://mirrors.aliyun.com/pypi/simple/ . Tested fallback: https://pypi.tuna.tsinghua.edu.cn/simple/ . For an individual failed download override PIP_INDEX_URL with the fallback, retaining constraints. Do not add multiple extra indexes or disable TLS. Cache roots are on the data disk.

Official ResNet18 weights are cached under cache/torch/hub/checkpoints; hash and actual GPU forward receipt are recorded. Existing UNI embeddings do not require downloading UNI weights. No unofficial Hugging Face endpoint configured. Future gated UNI downloads may require an account, only when online extraction is needed.

health_check.json records software/CUDA checks only, not algorithm efficacy. No model training or formal evaluation has been run.
