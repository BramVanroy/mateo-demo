# Format source code automatically
style:
	black src/mateo_st tests scripts
	isort src/mateo_st tests scripts

# Control quality
quality:
	black --check --diff src/mateo_st tests scripts
	isort --check-only src/mateo_st tests scripts
	flake8 src/mateo_st tests scripts
