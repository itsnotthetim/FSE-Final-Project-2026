TORCH_INDEX := https://download.pytorch.org/whl/cpu
BUILD_VENV := /tmp/build-venv
TRY_VENV := /tmp/try-venv
TEST_DIR := /tmp/proj-tests

.PHONY: wheel install test clean

$(BUILD_VENV)/bin/python:
	python3 -m venv $(BUILD_VENV)
	$(BUILD_VENV)/bin/pip install torch --index-url $(TORCH_INDEX)

$(TRY_VENV)/bin/python:
	python3 -m venv $(TRY_VENV)
	$(TRY_VENV)/bin/pip install torch --index-url $(TORCH_INDEX)

$(TRY_VENV)/bin/pytest: $(TRY_VENV)/bin/python
	$(TRY_VENV)/bin/pip install pytest

wheel: $(BUILD_VENV)/bin/python
	PIP_EXTRA_INDEX_URL=$(TORCH_INDEX) $(BUILD_VENV)/bin/pip wheel . --no-deps -w dist

install: wheel $(TRY_VENV)/bin/python
	$(TRY_VENV)/bin/pip install --force-reinstall dist/*.whl

test: install $(TRY_VENV)/bin/pytest
	rm -rf $(TEST_DIR) && cp -r tests $(TEST_DIR)
	cd /tmp && $(TRY_VENV)/bin/python -m pytest -q $(TEST_DIR)

clean:
	rm -rf dist build src/*.egg-info
	find . -name __pycache__ -prune -exec rm -rf {} +
