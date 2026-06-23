# Makefile for superbit-lensing
# Usage: make install

# Load environment variables from .env if it exists
-include .env

# Variables
PYTHON := python3
PIP := pip
CONDA := conda
ENV_NAME ?= superbit-env  # Default name, can be overridden
PYTHON_VERSION ?= 3.11    # Default Python version
SCAMP_VERSION ?= 2.14.0
SHELL := /bin/bash

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: all install clean env conda-deps pip-deps git-deps dev-install help check-conda

# Default target
all: install

# Help target
help:
	@echo "Available targets:"
	@echo "  make install     - Full installation (creates conda env, installs all deps, installs package)"
	@echo "  make env         - Create conda environment only"
	@echo "  make clean       - Clean build artifacts and cache"
	@echo "  make uninstall   - Remove conda environment"
	@echo "  make dev-install - Install in development mode"
	@echo ""
	@echo "To specify a custom environment name:"
	@echo "  make install ENV_NAME=myenv"
	@echo "  make uninstall ENV_NAME=myenv"
	@echo ""
	@echo "Or create a .env file from .env.example to set defaults"
	@echo ""
	@echo "Current settings:"
	@echo "  Environment name: $(ENV_NAME)"
	@echo "  Python version: $(PYTHON_VERSION)"

# Check if conda is installed
check-conda:
	@which conda > /dev/null || (echo "$(RED)Error: conda not found. Please install conda first.$(NC)" && exit 1)

# Create conda environment
env: check-conda
	@printf "$(GREEN)Creating conda environment: $(ENV_NAME)$(NC)\n"
	@conda env list | grep -q "^$(ENV_NAME) " || conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y
	@printf "$(GREEN)Environment created. Activate with: conda activate $(ENV_NAME)$(NC)\n"

# Install conda dependencies
conda-deps: env
	@printf "$(GREEN)Installing conda dependencies...$(NC)\n"
	@eval "$$(conda shell.bash hook)" && \
	conda activate $(ENV_NAME) && \
	conda install -c conda-forge \
		"astromatic-psfex=3.24.2" \
		"astromatic-source-extractor=2.28.0" \
		"astromatic-swarp=2.38.0" \
		-y && \
	SCAMP_PLATFORM="$$(conda info --json | $(CONDA) run -n $(ENV_NAME) python -c 'import json, sys; print(json.load(sys.stdin)["subdir"])')" && \
	if [ "$$SCAMP_PLATFORM" = "osx-arm64" ]; then \
		printf "%b%b" \
			"$(YELLOW)Skipping astromatic-scamp=$(SCAMP_VERSION) on $$SCAMP_PLATFORM because conda-forge does not publish that package for this platform.\n" \
			"Install SCAMP separately on a supported conda platform such as linux-64 or osx-64 if you need SCAMP-dependent workflows.$(NC)\n"; \
	else \
		conda install -c conda-forge "astromatic-scamp=$(SCAMP_VERSION)" -y; \
	fi

# Install pip dependencies
pip-deps: conda-deps
	@printf "$(GREEN)Installing pip dependencies...$(NC)\n"
	@eval "$$(conda shell.bash hook)" && \
	conda activate $(ENV_NAME) && \
	$(PIP) install --upgrade pip setuptools wheel && \
	bash -c '$(PIP) install esutil TreeCorr || \
		(conda install -c conda-forge TreeCorr -y && \
		 conda install -c conda-forge esutil -y)' && \
	bash -c '$(PIP) install --upgrade \
		pyyaml>=5.4 astropy fitsio piff shapely Rtree psutil \
		galsim>=2.3 lenspack pyregion numba>=0.57.0 pympler>=1.0 \
		numpy>=1.20 scipy>=1.7 matplotlib>=3.5 tqdm>=4.60 \
		seaborn>=0.11 astroquery ipdb statsmodels reproject colossus healpy || \
		CC=gcc CXX=c++ FC=gfortran $(PIP) install --no-cache-dir \
		pyyaml>=5.4 astropy fitsio piff shapely Rtree psutil \
		galsim>=2.3 lenspack pyregion numba>=0.57.0 pympler>=1.0 \
		numpy>=1.20 scipy>=1.7 matplotlib>=3.5 tqdm>=4.60 \
		seaborn>=0.11 astroquery ipdb statsmodels reproject colossus healpy'

# Install git dependencies
git-deps: pip-deps
	@printf "$(GREEN)Installing git dependencies...$(NC)\n"
	@eval "$$(conda shell.bash hook)" && \
	conda activate $(ENV_NAME) && \
	( $(PIP) install git+https://github.com/esheldon/ngmix.git --no-build-isolation \
	&& $(PIP) install git+https://github.com/esheldon/meds.git --no-build-isolation \
	&& $(PIP) install git+https://github.com/esheldon/psfex.git --no-build-isolation \
	|| CC=gcc CXX=c++ FC=gfortran $(PIP) install --no-cache-dir --no-build-isolation \
		git+https://github.com/esheldon/ngmix.git \
		git+https://github.com/esheldon/meds.git \
		git+https://github.com/esheldon/psfex.git )

# Install the package
install: git-deps
	@printf "$(GREEN)Installing superbit-lensing...$(NC)\n"
	@eval "$$(conda shell.bash hook)" && \
	conda activate $(ENV_NAME) && \
	$(PIP) install -e . || \
	CC=gcc CXX=c++ FC=gfortran $(PIP) install -e .
	@rm -f =*
	@printf "$(GREEN)Installation complete!$(NC)\n"
	@printf "$(YELLOW)To use the environment, run: conda activate $(ENV_NAME)$(NC)\n"
	@read -p "Do you want to run post-installation configuration? (y/n): " yn && \
	if [ "$$yn" = "y" ]; then \
	    printf "\n$(GREEN)Running post-installation configuration...$(NC)\n"; \
	    eval "$$(conda shell.bash hook)" && \
	    conda activate $(ENV_NAME) && \
	    $(PYTHON) post_installation.py --env_name $(ENV_NAME); \
	else \
	    printf "$(YELLOW)Skipping post-installation configuration.$(NC)\n"; \
	fi

# Development install (editable)
dev-install: git-deps
	@echo "$(GREEN)Installing superbit-lensing in development mode...$(NC)"
	@eval "$$(conda shell.bash hook)" && \
	conda activate $(ENV_NAME) && \
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Development installation complete!$(NC)"

# Clean build artifacts
clean:
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	# Clean up accidentally created files from version specs
	rm -f "=*" 2>/dev/null || true

# Uninstall (remove conda environment)
uninstall: check-conda
	@echo "$(YELLOW)Removing conda environment: $(ENV_NAME)$(NC)"
	@conda env remove -n $(ENV_NAME) -y
	@echo "$(GREEN)Environment removed.$(NC)"

# Test installation
test-install:
	@eval "$$(conda shell.bash hook)" && \
	conda activate $(ENV_NAME) && \
	python -c "import superbit_lensing; print('superbit-lensing successfully imported!')" && \
	python -c "import galsim; print('galsim successfully imported!')" && \
	python -c "import ngmix; print('ngmix successfully imported!')"
