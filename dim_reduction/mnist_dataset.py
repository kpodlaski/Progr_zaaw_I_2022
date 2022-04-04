import math

import matplotlib.pyplot as plt
import numpy as np
import torch as torch
import torchvision as torchvision
from sklearn.decomposition import PCA

def plot_sample(imgs, title, filename=None):
    plt.figure(figsize=(10, 4))
    plt.suptitle(title)
    cols = 5
    rows = math.ceil(len(imgs)/cols)
    for i in range(len(imgs)):
        ax = plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i].reshape(28,28), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if filename is not None :
        plt.savefig(filename)
    plt.show()
    plt.close()


mnist_data = torchvision.datasets.MNIST('../data/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
train_loader = torch.utils.data.DataLoader( mnist_data,  batch_size= len(mnist_data), shuffle=True)


examples = next(iter(train_loader))[0].numpy()
print(examples.shape)

mnist = examples.reshape(60000, 28*28)
print(mnist.shape)

sample = np.random.randint(mnist.shape[0],size = 20)
#sample = [24894, 17966, 54780, 40918, 42048,   980,  4966, 52147,  6204, 34810]
sample_img = [examples[x] for x in sample]

plot_sample(sample_img, "Original mnist images", "../data/out/mnist_orig.png")

pca = PCA(n_components=4)
red_mnist = pca.fit_transform(mnist)
print("d=4",pca.explained_variance_ratio_)
print("d=4",sum(pca.explained_variance_ratio_))
print("d=4",pca.components_)
print("d=4",pca.explained_variance_)
red_sample = [ red_mnist[x] for x in sample]
red_recreated = pca.inverse_transform(red_sample)
plot_sample(red_recreated, "PCA d=4", "../data/out/mnist_pca_4.png")



pca = PCA(n_components=128)
red_mnist = pca.fit_transform(mnist)
print("d=128",sum(pca.explained_variance_ratio_))
red_sample = [ red_mnist[x] for x in sample]
red_recreated = pca.inverse_transform(red_sample)
plot_sample(red_recreated, "PCA d=128", "../data/out/mnist_pca_128.png")

pca = PCA(n_components=.95)
red_mnist = pca.fit_transform(mnist)
print("d=0.95",sum(pca.explained_variance_ratio_))
print(len(pca.explained_variance_ratio_))
red_sample = [ red_mnist[x] for x in sample]
red_recreated = pca.inverse_transform(red_sample)
plot_sample(red_recreated, "PCA d=.95", "../data/out/mnist_pca_0.95.png")


