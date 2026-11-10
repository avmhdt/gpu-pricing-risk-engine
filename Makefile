# Task runner (§4 Tooling). Python commands run through uv; set UV_PROJECT_ENVIRONMENT to keep
# the venv on the WSL ext4 filesystem (e.g. ~/build/gpu-pricing-risk-engine/venv).
SHELL := /bin/bash
UV ?= uv
BUILD_DIR ?= $(HOME)/build/gpu-pricing-risk-engine
CXX_COMPILER ?= g++-13
CLANG_FORMAT ?= clang-format-18
CLANG_TIDY ?= clang-tidy-18
PYTEST_CPU := -m "not gpu and not bench and not data"
CPP_SOURCES := cpp/include/gpre/*.hpp cpp/tests/*.cpp src/gpre/cuda/kernels/*.cuh src/gpre/cuda/kernels/*.cu

.PHONY: setup setup-gpu lint type test test-gpu cpp cpp-asan cpp-tsan env data-smoke validate \
        bench profile risk report all fixtures verify-ac-w-01 verify-ac-w-06 verify-ac-w-12 verify-ac-w-13

setup:
	$(UV) sync --group dev

setup-gpu:
	$(UV) sync --group dev --group gpu

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(CLANG_FORMAT) --dry-run --Werror $(CPP_SOURCES)

type:
	$(UV) run mypy

test:
	JAX_ENABLE_X64=1 $(UV) run pytest $(PYTEST_CPU) --cov=gpre --cov-report=term --cov-fail-under=85

test-gpu:
	JAX_ENABLE_X64=1 $(UV) run pytest -m gpu --cov=gpre --cov-report=term

cpp:
	cmake -S cpp -B $(BUILD_DIR)/cpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(CXX_COMPILER)
	cmake --build $(BUILD_DIR)/cpp -j 8
	ctest --test-dir $(BUILD_DIR)/cpp --output-on-failure
	$(CLANG_TIDY) -p $(BUILD_DIR)/cpp cpp/tests/test_philox.cpp

cpp-asan:
	cmake -S cpp -B $(BUILD_DIR)/cpp-asan -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(CXX_COMPILER) -DGPRE_SANITIZE=address,undefined
	cmake --build $(BUILD_DIR)/cpp-asan -j 8
	ctest --test-dir $(BUILD_DIR)/cpp-asan --output-on-failure

cpp-tsan:
	cmake -S cpp -B $(BUILD_DIR)/cpp-tsan -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(CXX_COMPILER) -DGPRE_SANITIZE=thread
	cmake --build $(BUILD_DIR)/cpp-tsan -j 8
	TSAN_OPTIONS=halt_on_error=1 ctest --test-dir $(BUILD_DIR)/cpp-tsan --output-on-failure

env:
	$(UV) run gpre env

fixtures:
	$(UV) run python scripts/make_fixtures.py

# AC-W-13: FRED and Sharadar (or the local Stooq snapshot) ingest, then the strict data check.
data-smoke:
	$(UV) run gpre ingest fred
	$(UV) run gpre ingest sharadar || $(UV) run gpre ingest stooq
	$(UV) run gpre data check --strict

verify-ac-w-06:
	$(UV) run pytest tests/unit/test_philox.py tests/property/test_philox_properties.py
	$(MAKE) cpp

verify-ac-w-12: lint type test cpp cpp-asan cpp-tsan
	test -z "$$(git ls-files data)"

verify-ac-w-13: data-smoke

# GPU backends are included when a CUDA device is present (cloud instance); CPU-only otherwise.
VALIDATE_BACKENDS ?= numpy,jax$(shell $(UV) run python -c "import cupy" >/dev/null 2>&1 && echo ,cuda)

validate:
	$(UV) run gpre validate --suite rng
	JAX_ENABLE_X64=1 $(UV) run gpre validate --suite european --backends $(VALIDATE_BACKENDS)

verify-ac-w-01:
	JAX_ENABLE_X64=1 $(UV) run gpre validate --suite european --backends $(VALIDATE_BACKENDS)

bench profile risk report all:
	@echo "make $@ arrives in a later milestone (SPEC section 13)"; exit 1
