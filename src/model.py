import tensorflow as tf

class DeepRecommendationModel(tf.keras.Model):
    def __init__(self, num_users, num_events, embedding_size=64):
        super(DeepRecommendationModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.event_embedding = tf.keras.layers.Embedding(num_events, embedding_size)
        self.dense_layers = [tf.keras.layers.Dense(32, activation='relu') for _ in range(3)]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        event_vector = self.event_embedding(inputs[:, 1])
        x = tf.concat([user_vector, event_vector], axis=1)
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)