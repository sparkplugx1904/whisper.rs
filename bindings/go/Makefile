BUILD_DIR := build_go
MODELS_DIR := models
EXAMPLES_DIR := $(wildcard examples/*)
INCLUDE_PATH := $(abspath ../../include)
LIBRARY_PATH := $(abspath ../../target/release)

all: clean whisper examples

whisper: mkdir
	cargo build --manifest-path ../../Cargo.toml --package whisper-ffi --release

test: whisper modtidy
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} LD_LIBRARY_PATH=${LIBRARY_PATH} go test -v .

examples: $(EXAMPLES_DIR)

$(EXAMPLES_DIR): mkdir whisper modtidy
	@echo Build example $(notdir $@)
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go build -o ${BUILD_DIR}/$(notdir $@) ./$@

mkdir:
	@install -d ${BUILD_DIR}
	@install -d ${MODELS_DIR}

modtidy:
	@go mod tidy

clean:
	@echo Clean
	@rm -fr $(BUILD_DIR)
	@go clean
