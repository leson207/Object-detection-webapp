import tensorflow as tf
import joblib

from src.entity.configuration import ConfigManager
from src.components.DataIngestor import DataIngestor
from src.components.Preprocessor import Preprocessor
from src.components.DataModule import DataModule
from src.components.Trainer import Trainer
from src.components.Evaluator import Evaluator

if __name__=="__main__":
    config_manager = ConfigManager()

    # data_ingestor_config=config_manager.get_data_ingestor_config()
    # data_ingestor=DataIngestor(data_ingestor_config)
    # data_ingestor.prepare()
    # data_ingestor.sync_to_s3()
    # data_ingestor.sync_from_s3()

    # preprocessor_config=config_manager.get_preprocessor_config()
    # preprocessor=Preprocessor(preprocessor_config)
    # preprocessor.save()
    # joblib.load(preprocessor.config.ARTIFACT_DIR + '/train_transform.pkl')
    # joblib.load(preprocessor.config.ARTIFACT_DIR + '/test_transform.pkl')

    # data_module_config = config_manager.get_data_module_config()
    # data_module = DataModule(data_module_config)
    # data_loader=data_module.get_dataloader('train', 4)
    # data_loader=data_module.get_dataloader('validation', 4)
    # data_loader=data_module.get_dataloader('test', 4)
    # batch=next(iter(data_loader))
    # print(batch[0]['image'].shape)
    # print(batch[1]['target_1'].shape)
    # print(batch[1]['target_2'].shape)
    # print(batch[1]['target_3'].shape)

    # trainer_config = config_manager.get_trainer_config()
    # trainer = Trainer(trainer_config)
    # trainer.train()

    # evaluator_config = config_manager.get_evaluator_config()
    # evaluator = Evaluator(evaluator_config)
    # evaluator.eval()


    # model = tf.keras.models.load_model('artifacts/model/model.keras')
    model = tf.saved_model.load('artifacts/model/saved_model/1').signatures['serving_default']
    print(model.variables)

    print(model.summary())