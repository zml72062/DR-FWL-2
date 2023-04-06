# DR-FWL-2

## Dependencies

Code within this repository depends on `pygmmpp` package, which provides simple preprocessing API for graph datasets. Run the following command to install `pygmmpp`:

```
git clone https://github.com/zml72062/pygmmpp
cd pygmmpp
make
```

To run the code on Substructure Counting dataset, one must first run `make` under directory `./software/cycle_count` to compile the C code into `.so` binary. After that one can directly import `counting_dataset.py` to get the dataset. Notice that *there may be issues associated with ABI compatibility*, and we only tested our program on x86-64 Linux and MacOS platforms.

## Counting dataset

To run the code on Substructure Counting dataset, one can run
```
python count_model.py --copy-data
```

This will copy the raw data and configure file into a new directory, preprocess data in that directory, and run the model.

/05/2023: 
1.Slightly revise the preprocessing, now the preprocessing will not compute initial feature for 1-hop/2-hop edge. This part is done in the model right now.
2.Make zinc script runnable:
```
python train_zinc.py
```
