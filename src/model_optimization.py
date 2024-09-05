import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

class ModelOptimizer:
    def __init__(self, model_path):
        self.model_path = model_path

    def optimize_for_inference(self):
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP16,
            maximum_cached_engines=1000
        )
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=self.model_path,
            conversion_params=conversion_params)
        converter.convert()
        converter.save(output_saved_model_dir='./optimized_model')
        return './optimized_model'

    @staticmethod
    def load_optimized_model(optimized_model_path):
        return tf.saved_model.load(optimized_model_path)