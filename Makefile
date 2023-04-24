# Format source code automatically
style:
	black --line-length 119 --target-version py38 src/mateo_st
	isort src/mateo_st

# Control quality
quality:
	black --check --line-length 119 --target-version py38 src/mateo_st
	isort --check-only src/mateo_st
	flake8 src/mateo_st
