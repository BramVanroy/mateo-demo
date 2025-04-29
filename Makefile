quality:
	ruff check src/mateo_st/ scripts/ tests/
	ruff format --check src/mateo_st/ scripts/ tests/

style:
	ruff check src/mateo_st/ scripts/ tests/ --fix
	ruff format src/mateo_st/ scripts/ tests/
