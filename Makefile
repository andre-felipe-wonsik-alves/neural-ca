PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip


.PHONY: all setup install freeze run venv

all: run

setup:
	@${PYTHON} -m venv .venv
	@$(PIP) install pip
	@$(PIP) install -r requirements.txt
	@mkdir ./data ./output
	@echo "Now, you can run the simulation!"

venv:
	@source .venv/bin/activate.fish

install:
	@echo "Installing packages ;D"
	@$(PIP) install -r requirements.txt

freeze:
	@$(PIP) freeze > requirements.txt

run: install
	@echo "Packages verified!"
	@echo "Executing..."
	@$(PYTHON) main.py