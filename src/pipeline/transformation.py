import joblib

from src.entity.configuration import ConfigManager
from src.components.Preprocessor import Preprocessor

if __name__=="__main__":
    config_manager = ConfigManager()

    preprocessor_config=config_manager.get_preprocessor_config()
    preprocessor=Preprocessor(preprocessor_config)
    preprocessor.save()
    joblib.load(preprocessor.config.ARTIFACT_DIR + '/train_transform.pkl')
    joblib.load(preprocessor.config.ARTIFACT_DIR + '/test_transform.pkl')