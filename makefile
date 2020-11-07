install:
	make clean
	python setup.py install
build:
	python setup.py build
clean:
	rm -rf build/*
	rm -rf *egg*
	rm -rf dist
	rm -rf __pycache__
