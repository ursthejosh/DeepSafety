import numpy as np
import tensorflow as tf
import time
import datetime
from tensorboard import program

def main():

    model_name = "1686009518" # Insert Name of the model to relearn


    import_path = "./tmp/saved_models/{}".format(int(model_name))
    model = tf.keras.models.load_model(import_path)

    data_root = "./data/Train/"

    batch_size = 32
    img_height = 96  # 96 pixels for imagenet_mobilenet_v2_100_96, 224 pixels for mobilenet_v2 and inception_v3
    img_width = 96  # 96 pixels for imagenet_mobilenet_v2_100_96, 224 pixels for mobilenet_v2 and inception_v3

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    # check whether all your classes have been loaded correctly @class_names ['0' '1' '10' '11' '12']
    class_names = np.array(train_ds.class_names)
    print(class_names)

    # Preprocessing as the tensorflow hub models expect images as float inputs [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(x), y)
    )  # Where x—images, y—labels.
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y)
    )  # Where x—images, y—labels.

    # Then we set up prefetching will just smooth your data loader pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # //////////////////////////////////////// Preparing the model or heating up the coffee machine


    model.summary() #(in case you care how the whole thing looks now)

    # //////////////////////////////////////// Training or wild hand waving on caffeine

    # This starts tensorboard to you can check how your training is progressing
    # Helping you with tracking your training, resort to tensorboard, which can be accessed via the browser
    tracking_address = "./logs"
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    # This is stuff you are free to play around with
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )  # Enable histogram computation for every epoch.

    NUM_EPOCHS = 100  # This is probably not enough

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=tensorboard_callback,
    )

    # Save your model for later use. Early enough you should think about a model versioning system
    # and which information you will need to link with the model when doing so
    t = time.time()

    export_path = "./tmp/saved_models/{}".format(int(t))
    model.save(export_path)
    