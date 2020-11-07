build:
	python setup.py build

install:
	python setup.py install
clean:
	rm -rf build/*
	rm -rf *egg*
	rm -rf dist
	rm -rf __pycache__
