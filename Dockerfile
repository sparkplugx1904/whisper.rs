FROM rust:1-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p whisper-cli -p whisper-server -p whisper-quantize

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/whisper-cli /usr/local/bin/
COPY --from=builder /app/target/release/whisper-server /usr/local/bin/
COPY --from=builder /app/target/release/whisper-quantize /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["whisper-server"]
