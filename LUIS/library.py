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
