FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev build-essential
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
WORKDIR /src
RUN python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN python3 -m pip wheel . --no-deps -w /wheels

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip
RUN python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
COPY --from=build /wheels /wheels
RUN python3 -m pip install --break-system-packages /wheels/*.whl
RUN python3 -m pip install --break-system-packages pytest
COPY tests /opt/tests
CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]
