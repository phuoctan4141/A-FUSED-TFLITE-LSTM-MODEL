# A FUSED TFLITE LSTM MODEL

Demo using Human Activity Recognition to convert to TFLITE.

Minium Android 5.0

## Exporting the model

```python
run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
STEPS = 100
INPUT_SIZE = 12
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_lstm"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
```

## Adding metadata to TensorFlow Lite models

Adding metadata to help us using model with class labels and descriptions of inputs/outputs. More [https://www.tensorflow.org/lite/convert/metadata]

```python
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
```

```python
# Creates model info.
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "Human Activity Recognition"
model_meta.description = ("input[None, 100, 3] "
                          "is Sensor "
                          "output['Biking' ,' Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']")
model_meta.version = "v3.100.3"
model_meta.author = "phuoctan4141"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")
```

```python
# Creates input info.
input_meta = _metadata_fb.TensorMetadataT()

input_meta.name = 'inputSensor'
input_meta.description = (
    "Input is array data from sensor with 100steps".format(100, 3))
input_stats = _metadata_fb.StatsT()
input_meta.stats = input_stats
```

```python
labelmap_file = '/content/labelmap.txt'

export_model_path = '/content/model_LSTM_Metadata.tflite'
```

```python
import os
# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()

output_meta.name = "probability"
output_meta.description = "Probabilities of the 7 labels respectively."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename(labelmap_file)
label_file.description = "Labels for activities that the model can recognize."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]
```

```python
# Creates subgraph info.
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()
```

```python
# Pack metadata and associated files into the model
populator = _metadata.MetadataPopulator.with_model_file(export_model_path)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files([labelmap_file])
populator.populate()
```
