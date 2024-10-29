class Trainer:
    def __init__(self, data_module, training_config):
        self.train_dataset = data_module.get_dataloader('train',batch_size=training_config.batch_size)
        self.val_dataset = data_module.get_dataloader('validation',batch_size=training_config.batch_size)
        self.test_dataset = data_module.get_dataloader('test',batch_size=training_config.batch_size)

        if len(tf.config.list_physical_devices('GPU')) > 1:
            self.strategy = tf.distribute.MirroredStrategy()
        elif len(tf.config.list_physical_devices('GPU')) == 1:
            self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        else:
            self.strategy = tf.distribute.get_strategy()

        self.training_config=training_config
        with self.strategy.scope():
            self.model = YOLOMiniv2(3,208)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = training_config.learning_rate)
            self.loss_fn = YOLOLoss(reduction = tf.keras.losses.Reduction.NONE)
            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            self.val_dist_dataset = self.strategy.experimental_distribute_dataset(self.val_dataset)
            self.test_dist_dataset = self.strategy.experimental_distribute_dataset(self.test_dataset)

    def train(self):
        def forward(X, y, training=False):
            pred = self.model(X, training=training)
            loss = self.loss_fn(y, pred)
            loss = tf.reduce_sum(loss) * (1.0/self.training_config.batch_size)
            return loss, pred

        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                X, y = inputs
                with tf.GradientTape() as tape:
                    loss, pred = forward(X['image'], y, training=True)
                    # metrics()
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                return loss

            return _compute_metrics(step_fn, dist_inputs)

        @tf.function
        def val_step(dist_inputs):
            def step_fn(inputs):
                X, y = inputs
                loss, pred = forward(X['image'], y, training=False)
                # metric

                return loss

            return _compute_metrics(step_fn, dist_inputs)

        @tf.function
        def _compute_metrics(step_fn, dist_inputs):
            per_replica_loss = self.strategy.run(step_fn, args=(dist_inputs,))
            total_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            mean_loss = total_loss / tf.cast(self.training_config.batch_size, total_loss.dtype)
            return mean_loss

        with self.strategy.scope():
            num_train_step = tf.data.experimental.cardinality(self.train_dist_dataset).numpy()
            num_val_step = tf.data.experimental.cardinality(self.val_dist_dataset).numpy()

            for epoch in range(self.training_config.num_epochs):
                print(f'Epoch {epoch+1}/{self.training_config.num_epochs}')
                pbar = tf.keras.utils.Progbar(target=int(num_train_step), stateful_metrics=['loss'])
                metrics = {}
                for step, batch in enumerate(self.train_dist_dataset):
                    loss = train_step(batch)
                    metrics.update({'loss': loss.numpy()})
                    pbar.update(step+1, values=metrics.items(), finalize=False)

                for batch in self.val_dist_dataset:
                    loss = val_step(batch)
                    metrics.update({'val_loss': loss.numpy()})

                pbar.update(num_train_step, values=metrics.items(), finalize=True)

from types import SimpleNamespace

training_config=SimpleNamespace(learning_rate=1e-5,num_epochs=2,batch_size=2)
tmp=Trainer(DataModule(), training_config)
tmp.train()