from src.entity.configuration import ConfigManager
from src.components.Trainer import Trainer

if __name__=="__main__":
    config_manager = ConfigManager()

    trainer_config = config_manager.get_trainer_config()
    trainer = Trainer(trainer_config)
    trainer.train()