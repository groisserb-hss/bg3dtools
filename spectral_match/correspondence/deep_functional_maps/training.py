"""
Training loop for deep functional map networks.

Implements ensemble training with geodesic error supervision for
learning functional map correspondences between mesh pairs.
"""

import warnings

import numpy as np
from spectral_match.tools.util import get_platform
from spectral_match.tools.mesh_class import Mesh
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

import itertools

from . import operations as ops

"""====================================================================================="""
"""                                       Training                                      """
"""====================================================================================="""


def pairwise(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class EnsembleTrainer:
    def __init__(self, dataset, num_sigs, lr, bs, layers=7):
        self.func = ops.ResidualNet(layers, num_sigs, training=True)
        self.loss = tf.keras.metrics.Mean(name="train_loss")
        if get_platform() == 'darwin_arm64':
            self.optimiser = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        else:
            self.optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

        self.dataset = dataset.shuffle(buffer_size=128).batch(bs, drop_remainder=True)

    @tf.function
    def train_step(self, x, y):
        e_x, et_x, s_x, g_x = x
        e_y, et_y, s_y, g_y = y
        with tf.GradientTape() as tape:
            sigs = [self.func(x) for x in (s_x, s_y)]
            C = ops.correspondence_matrix(sigs, [et_x, et_y])
            P = ops.soft_correspondence_ensemble(C, et_x, e_y)
            loss = ops.geodesic_error_ensemble(P, g_x, g_y)
            grads = tape.gradient(loss, self.func.trainable_variables)
            self.optimiser.apply_gradients(
                zip(grads, self.func.trainable_variables)
            )
        self.loss(loss)

    def train(self, number_epochs, checkpoint_file=None):
        best = -1
        print_interval = int(number_epochs / 10)
        for epoch in range(number_epochs):
            for x, y in pairwise(self.dataset):
                try:
                    self.train_step(x, y)
                except Exception as e:
                    print(f"Error caught: {e}")
            epoch_loss = self.loss.result()
            if epoch % print_interval == 0:
                print("Epoch {}, Loss: {}".format(epoch + 1, 100 * epoch_loss))

            self.loss.reset_states()

            if epoch == 1:
                best = epoch_loss
            elif epoch_loss < best:
                if checkpoint_file is not None:
                    self.func.save_weights(checkpoint_file)
                best = epoch_loss

    def test(self):
        loss = []
        for x, y in pairwise(self.dataset):
            e_x, et_x, s_x, g_x = x
            e_y, et_y, s_y, g_y = y
            sigs = [self.func(x) for x in (s_x, s_y)]
            C = ops.correspondence_matrix(sigs, [et_x, et_y])
            P = ops.soft_correspondence_ensemble(C, et_x, e_y)
            loss = ops.geodesic_error_ensemble(P, g_x, g_y)
            loss.append(loss.numpy())
        return np.mean(loss)


def cache_and_train(raw_files: list[str], weight_file: str, mapper: 'FunctionalMapper',
                    lr=0.0002, batch_size=1, epochs=100, checkpoint_file=None):
    """
    NOTE THIS FUNCTION USES DEFAULT VALUES for num_wks, num_hks, num_gaussian, emin, emax, layers, lr, batch_size, epochs
    :param raw_files:
    :param weight_file:
    :return:
    """
    from tqdm import tqdm
    from os.path import isfile, join

    assert all([isfile(f) for f in raw_files]), 'Some files are missing'
    meshes = []
    for mf in tqdm(iterable=raw_files, desc='Loading/preprocessing meshes'):
        outfile = mf.replace('.ply', '.npz')
        if isfile(outfile):
            mesh = Mesh.from_file(outfile)
            assert mesh.s.shape[1] == 121
            meshes.append(mesh)
            continue

        mesh = Mesh.from_file(mf, normalize=True)
        mesh = mapper.preprocess_mesh(mesh.v, mesh.f)
        mesh.save_np(outfile)
        meshes.append(mesh)

    # prepare dataset
    evecs = [m.eigen[1].astype(np.float32) for m in meshes]
    evecs_t = [np.transpose(m.mass @ e).astype(np.float32) for e, m in zip(evecs, meshes)]
    sigs = [m.s.astype(np.float32) for m in meshes]
    metric = [m.g.astype(np.float32) for m in meshes]
    dataset = tf.data.Dataset.from_tensor_slices((evecs, evecs_t, sigs, metric))

    trainer = EnsembleTrainer(dataset, mapper.num_signatures, lr, batch_size, mapper.num_layers)
    trainer.train(epochs, checkpoint_file)
    resnet = trainer.func
    resnet.save_weights(weight_file)

    return resnet