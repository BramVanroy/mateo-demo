# Format source code automatically
style:
	black src/mateo_st
	isort src/mateo_st

# Control quality
quality:
	black --check --diff src/mateo_st
	isort --check-only src/mateo_st
	flake8 src/mateo_st
