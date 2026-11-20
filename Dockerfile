# SPDX-License-Identifier: Apache-2.0
# privacy-gate with the in-process encoder, serving the loopback sidecar on 8231.
#
#   docker build -t privacy-gate .
#   docker run --rm -p 127.0.0.1:8231:8231 privacy-gate
#   curl -s localhost:8231/check -d '{"text":"her biopsy is booked for the 20th"}'
#
# The encoder (BAAI/bge-m3, about 2.2 GB) is downloaded at build time, so the image runs with no network.
# The sidecar has no authentication: publish the port on 127.0.0.1 only, as above, or keep it inside a
# compose network. Inside the container it must bind 0.0.0.0 to be reachable at all; the port mapping is
# where the loopback rule is kept.
#
# Supply chain: the base image is pinned by digest, and the tag stays in front of the digest so Dependabot
# (docker ecosystem) can propose the next digest for the same tag. Python dependencies come from the exact
# pins in docker/requirements.txt (all of them, transitives included, installed with --no-deps). torch is
# installed first and on its own from the CPU-only wheel index, so the image never pulls CUDA wheels; its
# version is the `torch==` line of the same file, so there is one place to bump it.

# python:3.12-slim
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/opt/hf
WORKDIR /src
COPY docker/requirements.txt docker/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep -E '^torch==' docker/requirements.txt)" \
 && pip install --no-deps -r docker/requirements.txt \
 && python -c "import torch; assert torch.version.cuda is None, torch.version.cuda"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', device='cpu')"
COPY pyproject.toml README.md LICENSE NOTICE CHANGELOG.md ./
COPY model/head-v1.json model/
COPY src src
# The package itself has no dependencies; `[local]` is sentence-transformers, already installed above.
RUN pip install --no-deps . && pip check

# python:3.12-slim
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/privacy-gate /usr/local/bin/privacy-gate
COPY --from=build /opt/hf /opt/hf
RUN useradd --create-home --uid 10001 gate
# Numeric, so a runtime that checks for non-root (Kubernetes runAsNonRoot) can verify it without /etc/passwd.
USER 10001:10001
EXPOSE 8231
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8231/health', timeout=4)" || exit 1
ENTRYPOINT ["privacy-gate", "serve", "--backend", "local", "--host", "0.0.0.0", "--port", "8231"]
