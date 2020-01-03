import torch.utils.data

import numpy as np
import random

from src.dataloaders.dlbase import DataloaderFactory

class SimulationDataloaderFactory(DataloaderFactory):
    def __init__(self):
        super().__init__()

    def gettrainloader(self):
        if self.trainloader is None:
            train_set = SimDataset(self.config.dataloader_trainsize, transform=self._gettransform())
            self.trainloader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=self.config.dataloader_trainbatchsize,
                shuffle=False,
                num_workers=0)
        return self.trainloader

    def getvalloader(self):
        if self.valloader is None:
            val_set = SimDataset(self.config.dataloader_valsize, transform=self._gettransform())
            self.valloader = torch.utils.data.DataLoader(
                dataset=val_set,
                batch_size=self.config.dataloader_valbatchsize,
                shuffle=False,
                num_workers=0)
        return self.valloader

    def gettestloader(self):
        if self.testloader is None:
            test_set = SimDataset(self.config.dataloader_testsize, transform=self._gettransform())
            self.testloader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=self.config.dataloader_testbatchsize,
                shuffle=False,
                num_workers=0)
        return self.testloader


class SimDataset(torch.utils.data.Dataset):
    def __init__(self, count, transform=None):
        self.images, self.masks = self._generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image)
        return [image, mask]

    def _generate_random_data(self, height, width, count):
        x, y = zip(*[self._generate_img_and_mask(height, width) for i in range(0, count)])
        X = np.asarray(x) * 255
        X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
        Y = np.asarray(y)
        return X, Y

    def _generate_img_and_mask(self, height, width):
        '''
        MIT License

        Copyright (c) 2018 Naoto Usuyama

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        '''
        shape = (height, width)

        triangle_location = self._get_random_location(*shape)
        circle_location1 = self._get_random_location(*shape, zoom=0.7)
        circle_location2 = self._get_random_location(*shape, zoom=0.5)
        mesh_location = self._get_random_location(*shape)
        square_location = self._get_random_location(*shape, zoom=0.8)
        plus_location = self._get_random_location(*shape, zoom=1.2)

        # Create input image
        arr = np.zeros(shape, dtype=bool)
        arr = self._add_triangle(arr, *triangle_location)
        arr = self._add_circle(arr, *circle_location1)
        arr = self._add_circle(arr, *circle_location2, fill=True)
        arr = self._add_mesh_square(arr, *mesh_location)
        arr = self._add_filled_square(arr, *square_location)
        arr = self._add_plus(arr, *plus_location)
        arr = np.reshape(arr, (1, height, width)).astype(np.float32)

        # Create target masks
        masks = np.asarray([
            self._add_filled_square(np.zeros(shape, dtype=bool), *square_location),
            self._add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
            self._add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
            self._add_circle(np.zeros(shape, dtype=bool), *circle_location1),
            self._add_filled_square(np.zeros(shape, dtype=bool), *mesh_location),
            # _add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
            self._add_plus(np.zeros(shape, dtype=bool), *plus_location)
        ]).astype(np.float32)

        return arr, masks

    def _add_square(self, arr, x, y, size):
        s = int(size / 2)
        arr[x - s, y - s:y + s] = True
        arr[x + s, y - s: y + s] = True
        arr[x - s: x + s, y - s] = True
        arr[x - s:x + s, y + s] = True

        return arr

    def _add_filled_square(self, arr, x, y, size):
        s = int(size / 2)

        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

        return np.logical_or(arr, self._logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))

    def _logical_and(self, arrays):
        new_array = np.ones(arrays[0].shape, dtype=bool)
        for a in arrays:
            new_array = np.logical_and(new_array, a)

        return new_array

    def _add_mesh_square(self, arr, x, y, size):
        s = int(size / 2)

        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

        return np.logical_or(arr, self._logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))

    def _add_triangle(self, arr, x, y, size):
        s = int(size / 2)

        triangle = np.tril(np.ones((size, size), dtype=bool))

        arr[x - s:x - s + triangle.shape[0], y - s:y - s + triangle.shape[1]] = triangle  # pylint: disable=fixme, no-member

        return arr

    def _add_circle(self, arr, x, y, size, fill=False):
        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
        circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

        return new_arr

    def _add_plus(self, arr, x, y, size):
        s = int(size / 2)
        arr[x - 1:x + 1, y - s:y + s] = True
        arr[x - s:x + s, y - 1:y + 1] = True

        return arr

    def _get_random_location(self, width, height, zoom=1.0):
        x = int(width * random.uniform(0.1, 0.9))
        y = int(height * random.uniform(0.1, 0.9))

        size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

        return (x, y, size)
