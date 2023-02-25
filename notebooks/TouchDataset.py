import numpy as np

from functools import reduce
from operator import add


class TouchDataset:
    def __init__(self, filepath, noise_level=0, flatten=False, scope=None, *, tags=None):
        self.filepath = filepath
        self.noise_level = noise_level
        self.flatten = flatten
        self.scope = scope

        dataset = np.load(filepath, allow_pickle=True)
        if tags:
            self.tags = tags if isinstance(tags, list) else [tags]
        else:
            self.tags = list(dataset.keys())
        samples = reduce(add, [dataset[tag]["sensordata"] for tag in self.tags])
        orientations = reduce(add, [dataset[tag]["orientations"] for tag in self.tags])

        if noise_level > 0:
            for i, sample in enumerate(samples):
                samples[i] += np.random.normal(
                    np.mean(sample), noise_level * np.sqrt(np.var(sample)), sample.shape
                )
        self._shape = samples[0].shape
        self.samples = np.asarray(samples)
        self.orientations = np.asarray(orientations)

        if flatten:
            self.samples = self.samples.reshape(self.samples.shape[0], -1)
        if scope and scope[0] < scope[1]:
            for i, sample in enumerate(self.samples):
                if sample.max() > sample.min():
                    sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-12)
                    self.samples[i] = sample * (scope[1] - scope[0]) + scope[0]

    def subset(self, tags):
        """Create a new instance with only selected tags.

        Args:
            tags (str or list[str]): Tags to select.

        Returns:
            TouchDataset or None: A new instance with the same configs expect for tags.
        """
        if tags in self.tags:
            return self.__class__(
                self.filepath, self.noise_level, self.flatten, self.scope, tags=tags
            )
        return None

    def __getitem__(self, index):
        return self.samples[index], self.orientations[index]

    def __len__(self):
        return self.samples.shape[0]

    @property
    def shape(self):
        """Get the shape of the tactile data.

        Returns:
            tuple: Shape of the first sample.
        """
        return self._shape

    def split_set(self, ratio=0.5, shuffle=True):
        """Split samples to training set and test set by the ratio of len(training set)/len(self).

        Args:
            ratio (float, optional): len(training set)/len(self). Defaults to 0.5.
            shuffle (bool, optional): If samples and orientations need to be shuffled. Defaults to True.

        Returns:
            tuple of np.ndarray: X_train, y_train, X_test, y_test
        """
        arr = np.arange(len(self))
        if shuffle:
            np.random.shuffle(arr)
        shuffled_samples = self.samples[arr]
        shuffled_orientaions = self.orientations[arr]

        index = int(len(self) * ratio)
        return (
            shuffled_samples[:index],
            shuffled_orientaions[:index],
            shuffled_samples[index:],
            shuffled_orientaions[index:],
        )
