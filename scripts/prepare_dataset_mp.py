from config import FullConfig
from data_methods.dataset_processing import DatasetProcessing


full_config = FullConfig.load()
DatasetProcessing.generate_unique_compound_pools(full_config)
