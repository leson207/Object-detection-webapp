import tensorflow as tf

from src.entity.entity import TrainerConfig
from src.components.Criterion import YOLOLoss
from src.components.Model import YoloMimic
from src.components.DataModule import DataModule

from dataclasses import asdict

class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config=config

        data_module=DataModule(config.DATA_MODULE_CONFIG)
        self.train_dataset = data_module.get_dataloader(split='train',batch_size=config.TRAINING_CONFIG.BATCH_SIZE)
        self.val_dataset = data_module.get_dataloader(split='validation',batch_size=config.TRAINING_CONFIG.BATCH_SIZE)
        self.test_dataset = data_module.get_dataloader(split='test',batch_size=config.TRAINING_CONFIG.BATCH_SIZE)

        self.model = YoloMimic(**asdict(config.MODEL_CONFIG))
        # dummy_input=tf.random.normal([1, config.DATA_MODULE_CONFIG.IMAGE_SIZE, config.DATA_MODULE_CONFIG.IMAGE_SIZE, 3])
        # self.model(dummy_input)
        # self.model.build(input_shape=(None, config.DATA_MODULE_CONFIG.IMAGE_SIZE, config.DATA_MODULE_CONFIG.IMAGE_SIZE, 3))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = config.TRAINING_CONFIG.LEARNING_RATE)
        self.loss_fn = YOLOLoss(config.LOSS_CONFIG)

    def train(self):
        num_train_step = tf.data.experimental.cardinality(self.train_dataset).numpy()
        num_val_step = tf.data.experimental.cardinality(self.val_dataset).numpy()

        for epoch in range(self.config.TRAINING_CONFIG.NUM_EPOCHS):
            print(f'Epoch {epoch+1}/{self.config.TRAINING_CONFIG.NUM_EPOCHS}')
            pbar = tf.keras.utils.Progbar(target=int(num_train_step), stateful_metrics=['loss', 'val_loss'])
            metrics = {}
            for step, batch in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    pred = self.model(batch[0]['image'], training=True)
                    loss = self.loss_fn(batch[1], pred)

                # print(the_metrics.get_metrics(batch[1], pred))
                grads=tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


                metrics.update({'loss': loss.numpy()})
                pbar.update(step+1, values=metrics.items(), finalize=False)
                if step==10:
                    break

            for batch in self.val_dataset:
                pred = self.model(batch[0]['image'])
                loss = self.loss_fn(batch[1], pred)
                metrics.update({'val_loss': loss.numpy()})

            pbar.update(num_train_step, values=metrics.items(), finalize=True)

        self.model.save(self.config.ARTIFACT_DIR + '/model.keras')
        self.model.export(self.config.ARTIFACT_DIR + '/saved_model/1')