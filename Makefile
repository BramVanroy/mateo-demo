# Format source code automatically
style:
	black --line-length 119 --target-version py38 src/mateo_st
	isort src/mateo_st --line-length 119 --profile black

# Control quality
quality:
	black --check --line-length 119 --target-version py38 src/mateo_st --diff
	isort --check-only src/mateo_st --line-length 119 --profile black
	flake8 src/mateo_st
