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
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf
WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE CHANGELOG.md ./
COPY model/head-v1.json model/
COPY src src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[local]" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', device='cpu')"

FROM python:3.12-slim
ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/privacy-gate /usr/local/bin/privacy-gate
COPY --from=build /opt/hf /opt/hf
RUN useradd --create-home --uid 10001 gate
USER gate
EXPOSE 8231
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8231/health', timeout=4)" || exit 1
ENTRYPOINT ["privacy-gate", "serve", "--backend", "local", "--host", "0.0.0.0", "--port", "8231"]
