# Native ARM64 build; Basalt VIO has no CUDA or JetPack library dependency.
FROM ubuntu:24.04 AS builder
ARG RUST_VERSION=1.94.0
ARG BUILD_JOBS=4
ARG TARGET_CPU=cortex-a78
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain ${RUST_VERSION} \
    && rm /tmp/rustup.sh
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
COPY crates ./crates
COPY pipelines ./pipelines
COPY examples ./examples
COPY configs ./configs
RUN test "$(uname -m)" = aarch64 \
    && RUSTFLAGS="-C target-cpu=${TARGET_CPU}" cargo +${RUST_VERSION} build \
    --locked --release --jobs ${BUILD_JOBS} --example basalt_euroc_vio_demo --example basalt_stream_vio \
    --features basalt-lm-workspace-reuse \
    && strip target/release/examples/basalt_euroc_vio_demo target/release/examples/basalt_stream_vio

FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgcc-s1 python3 python3-numpy time ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/visloc-rs
COPY --from=builder /src/target/release/examples/basalt_euroc_vio_demo /usr/local/bin/
COPY --from=builder /src/target/release/examples/basalt_stream_vio /usr/local/bin/
COPY configs/basalt ./configs/basalt
COPY scripts/evaluate_euroc_trajectory.py ./scripts/
COPY docker/jetson/entrypoint.sh /usr/local/bin/visloc-euroc
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="visloc-rs Jetson Basalt VIO" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"
ENTRYPOINT ["visloc-euroc"]
CMD ["--help"]
