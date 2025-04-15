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

