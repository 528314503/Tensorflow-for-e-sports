import tensorflow as tf

class DistributedTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn

    def train(self, dataset, epochs):
        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs)
                loss = self.loss_fn(targets, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for inputs, targets in dataset:
                total_loss += self.strategy.run(train_step, args=(inputs, targets))
                num_batches += 1
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")