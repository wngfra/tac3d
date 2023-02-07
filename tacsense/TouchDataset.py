import numpy as np


class TouchDataset:
    def __init__(self, filepath, noise=True, flatten=False, normalize=None):
        dataset = np.load(filepath, allow_pickle=True)
        samples = (
            dataset["sharp_site"]["sensordata"]
            + dataset["round_site"]["sensordata"]
            + dataset["wedge_site"]["sensordata"]
        )
        orientations = (
            dataset["sharp_site"]["orientations"]
            + dataset["round_site"]["orientations"]
            + dataset["wedge_site"]["orientations"]
        )

        if noise:
            for i, sample in enumerate(samples):
                samples[i] += np.random.normal(
                    np.mean(sample), 0.5 * np.sqrt(np.var(sample)), sample.shape
                )
        self._shape = samples[0].shape
        self.samples = np.asarray(samples)
        self.orientations = np.asarray(orientations)

        if flatten:
            self.samples = self.samples.reshape(self.samples.shape[0], -1)
        if normalize is not None:
            samples = (self.samples - self.samples.min())/(self.samples.max() - self.samples.min())
            samples = samples*(normalize[1] - normalize[0]) + normalize[0]
            self.samples = samples

    def __getitem__(self, index):
        return self.samples[index], self.orientations[index]

    def __len__(self):
        return self.samples.shape[0]

    @property
    def shape(self):
        return self._shape

    def split_set(self, ratio=0.5):
        """Split samples to trainset and testset by the ratio=len(trainset)/len(self)"""
        arr = np.arange(len(self))
        np.random.shuffle(arr)
        index = int(len(self) * ratio)
        return (
            self.samples[:index],
            self.orientations[:index],
            self.samples[index:],
            self.orientations[index:],
        )
