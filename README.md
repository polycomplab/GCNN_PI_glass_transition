## Molecular property prediction toolkit

This toolkit can be used to train a GCNN to predict a molecular property (such as Glass Transition Temperature (Tg) or permeability) from SMILES description of a monomer molecule.

### "Synthetic" and experimental datasets of PolyAskInG database are available at http://polycomplab.org/index.php/ru/database.html

For the reference purpose and for the details of PolyAskIng database generation, please, see:

I.V. Volgin, P. Batyr, A.V. Matseevich, A.Y. Dobrovskiy, M.V. Andreeva, V.M. Nazarychev, S.V. Larin, M.Ya. Goikhman, Yu.V. Vizilter, A.A. Askadskii, S.V. Lyulin. 
Machine Learning with Enormous “Synthetic” Datasets: Predicting Glass Transition Temperature of Polyimides Using Graph Convolutional Neural Networks. 2022.

https://pubs.acs.org/doi/10.1021/acsomega.2c04649


## Examples
### Pretraining:
```
python3 main.py configs/config_pretrain_on_subset_perm.py
```

### Finetuning:
```
python3 main.py configs/config_finetune_perm.py
```