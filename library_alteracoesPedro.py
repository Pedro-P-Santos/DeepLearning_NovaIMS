import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.optimizers import AdamW

from sklearn.metrics import f1_score



def vis_images(train):
    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(np.argmax(labels[i])))
            plt.axis("off")
    plt.show()


def model_summary(model_class,input_shape=(224,224,3)):
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



def plot_training(history, title="Training History"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.suptitle(title)

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_f1_macro(model, test_dataset, name=""):
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1-score {name}: {f1_macro:.4f}")
    return f1_macro

def evaluate_f1_weighted(model, test_dataset, name=""):
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"Weighted F1-score {name}: {f1_weighted:.4f}")
    return f1_weighted




def fine_tune(model, fine_tune_at=50, lr=1e-5, base_model=None, base_layer_name="convnext_base", checkpoint_path="best_finetuned.keras", callbacks=None):
    if base_model is None:
        base_model = model.get_layer(base_layer_name)

    base_model.trainable = True

    if fine_tune_at > 0:
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False

    model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=3e-4),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    new_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)
    if callbacks:
        callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]
        callbacks.append(new_checkpoint)
    else:
        callbacks = [new_checkpoint]

    return model, callbacks