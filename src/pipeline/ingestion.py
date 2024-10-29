from src.entity.configuration import ConfigManager
from src.components.DataIngestor import DataIngestor

if __name__=="__main__":
    config_manager = ConfigManager()

    data_ingestor_config=config_manager.get_data_ingestor_config()
    data_ingestor=DataIngestor(data_ingestor_config)
    data_ingestor.prepare()
    # data_ingestor.sync_to_s3()
    # data_ingestor.sync_from_s3()