import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dropout,
    Conv2D,
    Flatten,
    Dense,
    MaxPooling2D,
)
from tensorflow.keras.utils import to_categorical

def generator(
    dir,
    gen=image.ImageDataGenerator(rescale=1.0 / 255),
    shuffle=True,
    batch_size=1,
    target_size=(24, 24),
    class_mode="categorical",
):
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode="grayscale",
        class_mode=class_mode,
        target_size=target_size,
    )

BS = 32
TS = (24, 24)
train_batch = generator("dataset_new/train", shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator("dataset_new/test", shuffle=True, batch_size=BS, target_size=TS)
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(SPE, VS)

cnn_model = Sequential(
    [
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(24, 24, 1)),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(4, activation="softmax"),
    ]
)
cnn_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
cnn_model.fit(
    train_batch,
    validation_data=valid_batch,
    epochs=50,
    steps_per_epoch=SPE,
    validation_steps=VS,
)      
accuracy = cnn_model.evaluate(valid_batch)[1]
print("Validation Accuracy:", accuracy)

cnn_model.save('models/cnnCat2.keras')

@tf.function(input_signature=[tf.TensorSpec(shape=(None, 24, 24, 1), dtype=tf.float32)])
def inference_func(input_image):
    return cnn_model(input_image)

concrete_func = inference_func.get_concrete_function(tf.TensorSpec(shape=(None, 24, 24, 1), dtype=tf.float32))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=cnn_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/cnncat2.tflite', 'wb') as f:
    f.write(tflite_model)