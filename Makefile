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
	black  baytune/ tests/ --check --config=./pyproject.toml
	ruff  baytune/ tests/ --config=./pyproject.toml

.PHONY: lint-fix
lint-fix:
	black  baytune/ tests/ --config=./pyproject.toml
	ruff  baytune/ tests/ --fix --config=./pyproject.toml