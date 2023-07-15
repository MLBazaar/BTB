.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete
	coverage erase

.PHONY: lint
lint:
	black baytune/ tests/ --check --config=./pyproject.toml
	ruff baytune/ tests/ --config=./pyproject.toml

.PHONY: lint-fix
lint-fix:
	black baytune/ tests/ --config=./pyproject.toml
	ruff baytune/ tests/ --fix --config=./pyproject.toml

.PHONY: installdeps-dev
installdeps-dev:
	python -m pip install ".[dev]"
	pre-commit install

.PHONY: installdeps-test
installdeps-test:
	python -m pip install ".[test]"

.PHONY: installdeps-docs
installdeps-docs:
	python -m pip install ".[docs]"

PYTEST = python -m pytest -s -vv -x
COVERAGE = --cov=baytune/ --cov-report term-missing --cov-config=./pyproject.toml --cov-report=xml:./coverage.xml

.PHONY: tests
tests:
	$(PYTEST) tests/

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: upgradesetuptools
upgradesetuptools:
	python -m pip install --upgrade setuptools

.PHONY: package
package: upgradepip upgradebuild upgradesetuptools
	python -m build
	$(eval PACKAGE=$(shell python -c 'import setuptools; setuptools.setup()' --version))
	tar -zxvf "dist/baytune-${PACKAGE}.tar.gz"
	mv "baytune-${PACKAGE}" unpacked_sdist

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='numpy|pandas|dill|scikit|dateutil|scipy|py|tornado|composeml|featuretools|matplotlib|tqdm')
	pip freeze | grep -v "BTB.git" | grep -E $(allow_list) > $(OUTPUT_FILEPATH)