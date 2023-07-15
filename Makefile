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
	black  btb/ tests/ --check --config=./pyproject.toml
	ruff  btb/ tests/ --config=./pyproject.toml

.PHONY: lint-fix
lint-fix:
	black  btb/ tests/ --config=./pyproject.toml
	ruff  btb/ tests/ --fix --config=./pyproject.toml