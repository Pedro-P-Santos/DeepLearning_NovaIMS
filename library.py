def vis_images(train):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(np.argmax(labels[i])))
            plt.axis("off")
    plt.show()


def model_summary(model_class,input_shape=(224,224,3)):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    """
    Prints the summary of the model.
    """
    model = model_class()
    inputs = Input(shape=input_shape)
    _ = model.call(inputs)
    model.summary()
    return model

def check_sizes(dataset):
    import numpy as np
    """
    Check the sizes of the images and labels in the dataset."""

    for images, labels in dataset.take(1):
        print("Image Batch Shape:", images.shape)
        print("Min Pixel Value:", np.min(images.numpy()))
        print("Max Pixel Value:", np.max(images.numpy()))
        print("Label Batch Shape:", labels.shape)



def data_loading(
    directory,
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    interpolation="bilinear",
    seed=42
):
    from tensorflow import keras

    return keras.utils.image_dataset_from_directory(
        directory=directory,
        labels=labels,
        label_mode=label_mode,
        batch_size=batch_size,
        image_size=image_size,
        color_mode=color_mode,
        shuffle=shuffle,
        interpolation=interpolation,
        seed=seed
    )