# Format source code automatically
style:
	black src/mateo_st tests
	isort src/mateo_st tests

# Control quality
quality:
	black --check --diff src/mateo_st tests
	isort --check-only src/mateo_st tests
	flake8 src/mateo_st tests
