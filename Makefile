PYTHON ?= python3
PYTHONPATH := $(CURDIR)/contracts/python:$(CURDIR)/Coll_Models_v2/src:$(CURDIR)/DSMC_0D_v2/src
export PYTHONPATH

.PHONY: all build test test-python smoke clean install

all: build test

build:
	$(MAKE) -C HS_CTC_v2/build all

test: test-python

test-python:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m unittest discover -s Coll_Models_v2/tests -v
	$(PYTHON) -m unittest discover -s DSMC_0D_v2/tests -v

smoke: build
	mkdir -p results/local_smoke
	cd HS_CTC_v2 && OMP_NUM_THREADS=1 ./build/SphCyl 0.8 1.0 1.0 1.0 ../results/local_smoke 123 32 v2
	$(PYTHON) HS_CTC_v2/scripts/finalize_run.py results/local_smoke
	@echo "CTC binary/schema smoke passed; coefficient recovery is covered by the synthetic estimator test."

install:
	$(PYTHON) -m pip install -e contracts -e Coll_Models_v2 -e DSMC_0D_v2

clean:
	$(MAKE) -C HS_CTC_v2/build clean
