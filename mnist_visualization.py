import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
train_mnist = datasets.MNIST('../mnist_data/',
                              download=True,
                              train=True)
for idx, (data, label) in enumerate(train_mnist):
    if idx > 0:
        break
    image = np.array(data)
    print(image)

    plt.figure()
    plt.imshow(data, cmap=plt.cm.Greys)
    plt.show()