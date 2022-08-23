import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional
from functools import partial

from dataJax2 import *
import optax

from evosax import PGPE


class DelaySequenceFitness(object):
    def __init__(
        self,
        # task_name: str = "SeqMNIST",
        batch_size: int = 128,
        # seq_length: int = 150,  # Sequence length in addition task
        # permute_seq: bool = False,  # Permuted S-MNIST task option
        test: bool = False,
        n_devices: Optional[int] = None,
    ):
        # self.task_name = task_name
        self.batch_size = batch_size
        self.steps_per_member = 1
        self.test = test
        self.loss_fn = loss_and_bce
        # self.loss_fn = loss_and_mae

        

        '''
        # Setup task-specific input/output shapes and loss fn
        if self.task_name == "SeqMNIST":
            self.action_shape = 10
            self.permute_seq = permute_seq
            self.seq_length = 784
            self.loss_fn = partial(loss_and_acc, num_classes=10)
        elif self.task_name == "Addition":
            self.action_shape = 1
            self.permute_seq = False
            self.seq_length = seq_length
            self.loss_fn = loss_and_mae
        else:
            raise ValueError("Dataset is not supported.")
        '''

        # data = get_array_data(
        #     self.task_name, self.seq_length, self.permute_seq, self.test
        # )

        self.dataloader = BatchLoader(batch_size=self.batch_size)

        # rng_input = jrandom.PRNGKey(42)
        # rng, rng_sample = jax.random.split(rng_input)
        # X, y = self.dataloader.sample(rng_sample)
        # print(X.shape,y.shape)

        # self.num_rnn_steps = self.dataloader.data_shape[1]
        self.num_rnn_steps = self.dataloader.rnn_step()  # change this
        print(self.num_rnn_steps)

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices

    def set_apply_fn(self, map_dict, network, carry_init):
        """Set the network forward function."""
        self.network = network
        self.carry_init = carry_init
        self.rollout_pop = jax.vmap(self.rollout_rnn, in_axes=(None, map_dict))
        # pmap over popmembers if > 1 device is available - otherwise pmap
        if self.n_devices > 1:
            self.rollout = self.rollout_pmap
            print(
                f"SequenceFitness: {self.n_devices} devices detected. Please"
                " make sure that the ES population size divides evenly across"
                " the number of devices to pmap/parallelize over."
            )
        else:
            self.rollout = jax.jit(self.rollout_vmap)

    def rollout_vmap(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ):
        """Vectorize rollout. Reshape output correctly."""
        loss, perf = self.rollout_pop(rng_input, network_params)
        loss_re = loss.reshape(-1, 1)
        perf_re = perf.reshape(-1, 1)
        return loss_re, perf_re

    def rollout_pmap(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ):
        """Parallelize rollout across devices. Split keys/reshape correctly."""
        keys_pmap = jnp.tile(rng_input, (self.n_devices, 1))
        loss_dev, perf_dev = jax.pmap(self.rollout_pop)(
            keys_pmap, network_params
        )
        loss_re = loss_dev.reshape(-1, 1)
        perf_re = perf_dev.reshape(-1, 1)
        return loss_re, perf_re

    def rollout_rnn(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ) -> Tuple[float, float]:
        """Evaluate a network on a supervised learning task."""
        rng, rng_sample = jax.random.split(rng_input)
        X, y = self.dataloader.sample(rng_sample)
        # Map over sequence batch dimension
        y_pred = jax.vmap(self.rollout_single, in_axes=(None, None, 0))(
            rng, network_params, X
        )
        loss, perf = self.loss_fn(y_pred, y)
        # Return negative loss to maximize!
        return -1 * loss, perf

    def rollout_single(
        self,
        rng: chex.PRNGKey,
        network_params: chex.ArrayTree,
        X_single: chex.ArrayTree,
    ):
        """Rollout RNN on a single sequence."""
        # Reset the network
        hidden = self.carry_init()

        def rnn_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            network_params, hidden, rng, t = state_input
            rng, rng_net = jax.random.split(rng)
            hidden, pred = self.network(
                network_params,
                X_single[t],
                hidden,
                rng_net,
            )
            carry = [network_params, hidden, rng, t + 1]
            return carry, pred

        # Scan over image length (784)/sequence
        _, scan_out = jax.lax.scan(
            rnn_step, [network_params, hidden, rng, 0], (), self.num_rnn_steps
        )
        y_pred = scan_out[-1]
        return y_pred

    @property
    def input_shape(self) -> Tuple[int]:
        """Get the shape of the observation."""
        return self.dataloader.data_shape


def loss_and_acc(
    y_pred: chex.Array, y_true: chex.Array, num_classes: int
) -> Tuple[chex.Array, chex.Array]:
    """Compute cross-entropy loss and accuracy."""
    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
    labels = jax.nn.one_hot(y_true, num_classes)
    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
    loss /= labels.shape[0]
    return loss, acc


def loss_and_mae(
    y_pred: chex.Array, y_true: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Compute mean squared error loss and mean absolute error."""
    loss = jnp.mean((y_pred.squeeze() - y_true) ** 2)
    mae = jnp.mean(jnp.abs(y_pred.squeeze() - y_true))
    return loss, -mae

def loss_and_bce(y_pred: chex.Array, y_true: chex.Array
) -> Tuple[chex.Array, chex.Array]:

    # print(type(y_pred),type(y_true))
    y_pred = y_pred.astype(float)

    # loss = -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))
    criterion = optax.sigmoid_binary_cross_entropy(y_pred,y_true)
    loss = jnp.mean(criterion)
    return loss



class BatchLoader:
    def __init__(
        self,
        # X: chex.Array,
        # y: chex.Array,
        batch_size: int,
):
        # self.X = X
        # self.y = y
        # self.data_shape = self.X.shape[1:][::-1]
        # self.num_train_samples = X.shape[0]
        self.batch_size = batch_size

    def sample(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        # print(key)
        key = jrandom.split(key, num=batch_size)
        vf = vmap(custom2)
        inputs,targets,masks = vf(key)
        # # print('vf is\n',inputs,targets,masks)

        return (inputs,targets)

    def rnn_step(self):
        key = jrandom.PRNGKey(42)
        X,y = self.sample(key)
        return X.shape[1]

    

ob = DelaySequenceFitness(batch_size = 128)