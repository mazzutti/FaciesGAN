.PHONY: run-default configure build clean rebuild

run-default:
	bash scripts/run_default_faciesgan.sh

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

build: configure
	cmake --build build -- -j4

clean:
	rm -rf build

rebuild: clean configure build
