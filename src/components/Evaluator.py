import joblib
import tensorflow as tf

from src.components.DataModule import DataModule
from src.components.Criterion import YOLOLoss
from src.entity.entity import EvaluatorConfig

class Evaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config=config
        data_module=DataModule(config.DATA_MODULE_CONFIG)
        self.test_dataset = data_module.get_dataloader(split='test',batch_size=config.BATCH_SIZE)

        self.loss_fn = YOLOLoss(config.LOSS_CONFIG)
        self.processor = joblib.load(self.config.PREPROCESSOR_PATH)
    
    def eval(self):
        model = tf.keras.models.load_model(self.config.KERAS_MODEL_PATH)
        self.single_eval(model)
        model = tf.saved_model.load(self.config.SAVED_MODEL_PATH).signatures['serving_default']
        self.single_eval(model,'saved_model')

    def single_eval(self, model, format=None):
        num_test_step = tf.data.experimental.cardinality(self.test_dataset).numpy()

        pbar = tf.keras.utils.Progbar(target=int(num_test_step), stateful_metrics=['test_loss'])
        metrics = {}
        for step, batch in enumerate(self.test_dataset):
            pred = model(batch[0]['image'])
            if format=='saved_model':
                pred={
                    0: pred['output_0'],
                    1: pred['output_1'],
                    2: pred['output_2']
                }
            loss = self.loss_fn(batch[1], pred)

            metrics.update({'test_loss': loss.numpy()})
            pbar.update(step+1, values=metrics.items(), finalize=False)
            if step==10:
                break

        pbar.update(num_test_step, values=metrics.items(), finalize=True)