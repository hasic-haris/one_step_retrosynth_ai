from config import FullConfig
from data_methods.dataset_processing import DatasetProcessing


full_config = FullConfig.load()

DatasetProcessing.generate_compound_pools_from_dataset(full_config)
