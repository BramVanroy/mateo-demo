# Format source code automatically
style:
	black --line-length 119 --target-version py38 mateo_st/
	isort mateo_st/ --line-length 119

# Control quality
quality:
	black --check --line-length 119 --target-version py38 mateo_st/
	isort --check-only --line-length 119 mateo_st/
	flake8 mateo_st/ --exclude __pycache__,__init__.py
