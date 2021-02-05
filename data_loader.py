import pandas as pd

dataset = pd.read_csv("data_source/data_processed.csv")[["rxn_smiles", "class"]]
print(dataset.head())
