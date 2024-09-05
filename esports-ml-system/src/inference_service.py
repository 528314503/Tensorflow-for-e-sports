import tensorflow as tf

class InferenceService:
    def __init__(self, model):
        self.model = model

    def recommend_events(self, user_ids, available_event_ids):
        user_event_pairs = tf.constant([(u, e) for u in user_ids for e in available_event_ids])
        scores = self.model(user_event_pairs)
        recommendations = {}
        for user_id in user_ids:
            user_scores = scores[user_event_pairs[:, 0] == user_id]
            top_events = tf.gather(available_event_ids, tf.argsort(user_scores, direction='DESCENDING'))
            recommendations[user_id] = top_events[:5].numpy().tolist()  # Top 5 recommendations
        return recommendations

    def predict_optimal_price(self, event_features):
        return self.model(event_features).numpy()