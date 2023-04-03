# DR-FWL-2

## Dependencies

Code within this repository depends on `pygmmpp` package, which provides simple preprocessing API for graph datasets. Run the following command to install `pygmmpp`:

```
git clone https://github.com/zml72062/pygmmpp
cd pygmmpp
make
```

To run the code on Substructure Counting dataset, one must first run `make` under directory `./software/cycle_count` to compile the C code into `.so` binary. After that one can directly import `counting_dataset.py` to get the dataset. Notice that *there may be issues associated with ABI compatibility*, and we only tested our program on x86-64 Linux and MacOS platforms.

