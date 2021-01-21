import pandas as pd
from tqdm import tqdm

from chemistry_methods.reaction_cores import get_reaction_core_atoms
from chemistry_methods.reactions import parse_reaction_roles

from rdkit.Chem import AllChem


def main():
    """
    dataset = pd.read_pickle("/data/hhasic/project_generated_output/one_step_retrosynth_ai/fold_1/training_data.pkl")[
        ["reaction_smiles", "reaction_class"]]

    i, p_smiles, core_atoms = 0, [], []

    for row_ind, row in tqdm(dataset.iterrows(), total=len(dataset.index), ascii=True, desc="Doing sum tricky"):
        _, _, ps = parse_reaction_roles(row["reaction_smiles"], as_what="canonical_smiles")
        _, _, ps_mol = parse_reaction_roles(row["reaction_smiles"], as_what="mol")

        #p_smiles.append(ps[0])
        ca = get_reaction_core_atoms(row["reaction_smiles"])[1][0]
        core_atoms.append(ca)

        for p in ps_mol[0].GetAtoms():
            if p.GetIdx() not in ca:
                p.SetAtomMapNum(0)
            else:
                p.SetAtomMapNum(1)

        p_smiles.append(AllChem.MolToSmiles(ps_mol[0], canonical=True))

    dataset["product_smiles"] = p_smiles
    dataset["core_atoms"] = core_atoms

    print(dataset.columns)
    print(dataset.index)
    print(dataset.head(5))

    dataset.to_csv("/data/hhasic/project_generated_output/train.csv", index=False)

    """
    tr = pd.read_csv("/data/hhasic/project_generated_output/train.csv")[["product_smiles", "reaction_class"]]
    val = pd.read_csv("/data/hhasic/project_generated_output/validation.csv")[["product_smiles", "reaction_class"]]
    tst = pd.read_csv("/data/hhasic/project_generated_output/test.csv")[["product_smiles", "reaction_class"]]

    final = pd.concat([tr, val, tst])

    print(final.shape)
    print(final.head(5))

    final.to_csv("/data/hhasic/project_generated_output/rxn_pred.csv", index=False)


if __name__ == "__main__":
    main()
