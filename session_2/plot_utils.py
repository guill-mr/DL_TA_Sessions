import numpy as np
import matplotlib.pyplot as plt

"""
In this notebook we have the boilerplate tools used to visualize the data.

Basic plots are the first thing everyone generates with AI completely ;)
I did tweak a couple of things but nothing fancy.
(It's useful to know a bit how plt works to tailor your functions, but that's not the point of this course).

Do notice that seed=42 gets overridden if we pass our SEED variable to the function.
"""

def plot_random_raw_samples(dataset, class_names, n: int = 16, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)

    side = int(np.ceil(np.sqrt(n)))
    if n>10:
        fig, axes = plt.subplots(side, side, figsize=(10, 10))
    elif n <= 10:
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    for ax in axes:
        ax.axis('off')

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        axes[i].imshow(np.array(image), cmap='gray')
        axes[i].set_title(class_names[label], fontsize=9)
        axes[i].axis('off')

    plt.suptitle('Random raw samples from FashionMNIST', fontsize=14)
    # plt.tight_layout()
    plt.show()


def plot_one_per_class(dataset, class_names) -> None:
    first_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in first_indices:
            first_indices[label] = idx
        if len(first_indices) == len(class_names):
            break

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    for class_id, class_name in enumerate(class_names):
        image, _ = dataset[first_indices[class_id]]
        axes[class_id].imshow(np.array(image), cmap='gray')
        axes[class_id].set_title(f'{class_id}: {class_name}')
        axes[class_id].axis('off')

    plt.suptitle('At least one raw example from each class', fontsize=14)
    plt.show()