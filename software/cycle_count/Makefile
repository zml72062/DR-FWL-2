all:
	gcc -o find_cycle.so -fPIC -shared -O2 find_cycle.c
	cd ..; python3 cycle_count/setup.py install

clean:
	rm -f find_cycle.so
	find . | grep -E "(/__pycache__|\.pyc|\.pyo)" | xargs rm -rf
