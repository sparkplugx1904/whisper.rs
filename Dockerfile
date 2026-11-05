FROM rust:1-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p whisper-cli -p whisper-server -p whisper-quantize

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/* \
    && useradd -u 10001 -U -m -s /bin/sh whisper
WORKDIR /app
COPY --from=builder /app/target/release/whisper-cli /usr/local/bin/
COPY --from=builder /app/target/release/whisper-server /usr/local/bin/
COPY --from=builder /app/target/release/whisper-quantize /usr/local/bin/
USER whisper
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
ENTRYPOINT ["whisper-server"]
