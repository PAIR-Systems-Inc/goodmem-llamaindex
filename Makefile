GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

test:	## Run unit tests (no live server required).
	pytest tests/test_tools_goodmem.py -v

test-e2e:	## Run live e2e smoke test (needs GOODMEM_* env vars).
	pytest tests/test_tools_goodmem_e2e.py -v

test-all:	## Run unit + e2e tests.
	pytest tests -v
