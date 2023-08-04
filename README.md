# DR-FWL-2

## Dependencies

Code within this repository depends on `pygmmpp` package (https://anonymous.4open.science/r/pygmmpp-4AF6), which provides simple preprocessing API for graph datasets. 

After downloading `pygmmpp` from the above URL, run `make` under the root directory to install the `pygmmpp` package.

Other requirements include:

* python 3.9.12
* numpy 1.21.5
* pytorch 1.11.0
* pytorch-scatter 2.0.9
* pytorch-sparse 0.6.14
* pytorch-geometric (pyg) 2.1.0
* pytorch-lightning 2.0.1
* wandb 0.14.0
* torchmetrics 0.11.4
* rdkit 2022.3.5
* ogb 1.3.3
* scikit-learn 1.1.1
* scipy 1.7.3
* h5py 3.7.0
* tqdm 4.64.0


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

To run SSWL/SSWL+/LFWL/SLFWL GNN on QM9, execute

```
python models_qm9.py --seed <random seed> --config-path configs/qm9.json --lfwl <name>
```
where `<name>` is `SSWL`/`SSWLPlus`/`LFWL`/`SLFWL`.

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

## Cycle counting on protein datasets

We collect three protein datasets from https://github.com/phermosilla/IEConv_proteins, two of which (`ProteinsDBDataset` and `HomologyTAPEDataset`) are used for cycle counting. See https://anonymous.4open.science/r/ProteinsDataset-F0C2 for our original code that processes the three datasets.

Download the two datasets from the following URLs:

* `ProteinsDBDataset`

https://drive.google.com/uc?export=download&id=1KTs5cUYhG60C6WagFp4Pg8xeMgvbLfhB

Extract in `protdb/raw/ProteinsDB/`

* `HomologyTAPEDataset`

https://drive.google.com/uc?export=download&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar

Extract in `homology/raw/HomologyTAPE/`

We copied the `IEProtLib` directory from https://github.com/phermosilla/IEConv_proteins since our processing code makes use of this submodule. We also copied code from https://github.com/GraphPKU/I2GNN (the official code for [Boosting the Cycle Counting Power of Graph Neural Networks with I $^2$-GNNs.](https://arxiv.org/abs/2210.13978)) to run baseline methods (MPNN, NGNN, I2GNN and PPGN) on the two proteins datasets.

**Experiments on the two protein datasets depend on the `h5py` package.**

To run 2-DRFWL(2) GNN/MPNN/NGNN/I2GNN/PPGN on `ProteinsDBDataset`, execute
```
python train_on_proteins.py --dataset ProteinsDB --model <model> --root protdb --target <target> --batch_size 32 --h 3 --cuda 0 --epochs 1500 --test_split <test-split>
```
where `<model>` takes `DRFWL2`/`MPNN`/`NGNN`/`I2GNN`/`PPGN`, `<target>` takes `3-cycle`/`4-cycle`/`5-cycle`/`6-cycle`, `<test-split>` takes 0-9 for 10-fold cross validation.

To run 2-DRFWL(2) GNN/MPNN/NGNN/I2GNN/PPGN on `HomologyTAPEDataset`, execute
```
python train_on_proteins.py --dataset HomologyTAPE --model <model> --root homology --target <target> --batch_size 32 --h 3 --cuda 0 --epochs 2000
```