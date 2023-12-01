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

### Prediction:
```
python3 scripts/predict.py --checkpoint_path "checkpoints/finetune/trained_network_weights.pth"
```
Create a file datasets/prediction/SMILES.csv and save a list of SMILES to it, one SMILES string per line.

Examples of SMILES.csv:
```
Ic1ccc(cc1)Oc1ccc(cc1)N1C(=O)c2c(C1=O)cc(cc2)Oc1cccc2c1cccc2Oc1ccc2c(c1)C(=O)N(C2=O)I
Ic1cccc(c1)N1C(=O)c2c(C1=O)cc(cc2)Oc1cccc2c1cccc2Oc1ccc2c(c1)C(=O)N(C2=O)I
```
Predicted Tg values will be saved to datasets/prediction/predictions.csv

Example of predictions.csv:
```
SMILES,"Tg, K, pred"
Ic1ccc(cc1)Oc1ccc(cc1)N1C(=O)c2c(C1=O)cc(cc2)Oc1cccc2c1cccc2Oc1ccc2c(c1)C(=O)N(C2=O)I,524.7864379882812
Ic1cccc(c1)N1C(=O)c2c(C1=O)cc(cc2)Oc1cccc2c1cccc2Oc1ccc2c(c1)C(=O)N(C2=O)I,535.5695190429688
```