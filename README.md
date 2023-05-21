# DR-FWL-2

## Dependencies

Code within this repository depends on `pygmmpp` package (https://anonymous.4open.science/r/pygmmpp-4AF6), which provides simple preprocessing API for graph datasets. 

After downloading `pygmmpp` from the above URL, run `make` under the root directory to install the `pygmmpp` package.

## Counting dataset

To run the code on Substructure Counting dataset, one must first run `make` under directory `./software/cycle_count` to compile the C code into `.so` binary and install the python module that generates the ground-truth for substructure counting. After that one can directly import `counting_dataset.py` to get the dataset. Notice that *there may be issues associated with ABI compatibility*, and we only tested our program on x86-64 Linux and MacOS platforms. Alternatively, one can also download from https://anonymous.4open.science/r/cycle_count-E817 and run `make` under the root directory of that repository.

To run 2-DRFWL(2) GNN on Substructure Counting dataset, one can run

```
python train_on_count.py --seed <random seed> --config-path configs/count.json
```

Training settings are saved in `configs/count.json` by default. **NOTICE** that every time you modify `dataset.target` setting in the configure file, you should delete the `processed` directory under `datasets/count` to preprocess the dataset again for another target. 

## ZINC

/05/2023: 
1.Slightly revise the preprocessing, now the preprocessing will not compute initial feature for 1-hop/2-hop edge. This part is done in the model right now.
2.Make ZINC script runnable:
```
python train_zinc.py
```

## QM9

To run 2-DRFWL(2) GNN on QM9, execute

```
python models_qm9.py --seed <random seed> --config-path configs/qm9.json
```

Training settings are saved in `configs/qm9.json` by default.

## EXP

To run 2-DRFWL(2) GNN on EXP dataset, execute

```
python run_exp.py --epochs <num of epochs>
```

## SR25

To run 2-DRFWL(2) GNN on SR25 dataset, execute

```
python run_sr.py --num-epochs <num of epochs>
```

## ogbg-molhiv

To run 2-DRFWL(2) GNN on ogbg-molhiv dataset, execute

```
python ogbg_molhiv_models.py --config-path configs/ogbmol.json
```
