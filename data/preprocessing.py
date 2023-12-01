import random
import time
import warnings

from rdkit import rdBase, Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torch
import torch_geometric


class Preprocessing:
    def __init__(self, cyclingbefore=False) -> None:
        self.cyclingbefore = cyclingbefore

    def graph_from_smiles(self, smiles):
        #### Change Iodine atom type to *
        smiles = list(smiles)
        for i, l in enumerate(smiles):
            if(l=="I"):
                smiles[i]="*"
        smiles = "".join(smiles)
        #####
        mol = Chem.MolFromSmiles(smiles)
        
        if self.cyclingbefore:
            mol = self._add_cyclic_connection(mol)
        
        #### Embedding
        mol = Chem.AddHs(mol)

        blocker = rdBase.BlockLogs()  # turns rdkit messages off
        # to suppress "UFFTYPER: Unrecognized atom type: *_"

        res = AllChem.EmbedMolecule(mol)
        if res == -1:  # -1 means that embedding has failed
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res == -1:
            raise RuntimeError('2D->3D molecule conversion has failed')
        t1 = time.time()
        while(AllChem.MMFFOptimizeMolecule(mol)==1):
            if time.time()-t1 > 10:  # timeout
                break
        del blocker  # turns rdkit messages back on
        
        mol = Chem.RemoveHs(mol)
        if not self.cyclingbefore:
            mol = self._add_cyclic_connection(mol)
        return self._graph_from_mol(mol)

    def _add_cyclic_connection(self, mol):
        stars = []
        nbs = []
        for j, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if atom_symbol == '*':
                bonds = list(atom.GetBonds())
                assert len(bonds) == 1
                stars.append(atom.GetIdx())
                bond_type = bonds[0].GetBondType()
                for a in atom.GetNeighbors():
                    nbs.append(a.GetIdx())
        edmol = Chem.EditableMol(mol)
        # Draw.MolToFile(mol,f'mol_{random.randrange(0, 100000000)}.png')
        try:
            edmol.AddBond(nbs[0],nbs[1],order=bond_type)
        except RuntimeError:
            print('bond already exists, skipping...')
        if (stars[0]>stars[1]):
            edmol.RemoveAtom(stars[0])
            edmol.RemoveAtom(stars[1])
        else:
            edmol.RemoveAtom(stars[1])
            edmol.RemoveAtom(stars[0])
        return edmol.GetMol()

    def _graph_from_mol(self, mol):
        #### Vertices data
        # atom_symbols = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'Si', '*')
        # NOTE when permeability pretraining (PA_syn_perm_He) only these atoms are encountered (in raw smiles):  ['C', 'N', 'O', 'F', 'S', 'I']
        atom_symbols = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'Si', 'P', 'Na', '*')
        x = torch.zeros([mol.GetNumAtoms(), len(atom_symbols) + 1], dtype=torch.float32)
        for j, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            try:
                idx = atom_symbols.index(atom_symbol)
            except ValueError as e:
                print('unexpected atom:', atom_symbol)
                raise e
            x[j, idx] = 1
            x[j, -1] = atom.GetExplicitValence() - atom.GetDegree()
        #### Edge data
        nTypes = 4
        bondTypes = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3,
            }
        edge_index = []
        edge_types = []
        # edge_cycle = []
        for bond in mol.GetBonds():
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_types.append(bondTypes[bond.GetBondType()])
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        ## Edge attributes
        nEdges = edge_index.shape[0]
        edgeType = torch.zeros(nEdges, nTypes)
        edgeType[torch.arange(nEdges), edge_types] = 1
        # import random
        # Dists
        # try:
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
        # except ValueError as e:
        #     Draw.MolToFile(mol,f'mol_{random.randrange(0, 100000000)}.png')
        #     # print(Chem.MolToSmiles(mol))
        #     raise e
        edgeLength = torch.norm(pos[edge_index[:, 0]] - pos[edge_index[:, 1]], p=2, dim=1)
        if not self.cyclingbefore:
            edgeLength[-1] = random.gauss(mu=1.39, sigma=0.3)  # TODO test
        # Cyclic info
        edgeCyc = torch.zeros(nEdges, 5)
        for i, bond in enumerate(mol.GetBonds()):
            for k in range(4, 9):
                if bond.IsInRingSize(k):
                    edgeCyc[k-4] = 1
        edgeAttrs = torch.cat([edgeType, edgeLength.unsqueeze(-1), edgeCyc], dim = 1)
        edgeAttrs = edgeAttrs.repeat(2, 1)
        edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0).contiguous()
        return torch_geometric.data.Data(x=x, edge_index=edge_index.t(), edge_attr=edgeAttrs, pos=pos)
