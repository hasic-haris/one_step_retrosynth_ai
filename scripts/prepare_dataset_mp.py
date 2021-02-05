from config import FullConfig
from data_methods.dataset_processing import DatasetProcessing


# Preparation: Read the configuration files.
full_config = FullConfig.load()

# Preparation: Create and initialize the DatasetProcessing object.
dataset_processing = DatasetProcessing(full_config)

print("\nStep 1/5: Generate unique compound pools for the reactants and products separately.\n")
#dataset_processing.generate_unique_compound_pools()

print("\nStep 2/5: Expand the original dataset with additional, useful information.\n")
dataset_processing.main()
