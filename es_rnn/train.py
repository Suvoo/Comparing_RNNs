import argparse
import os

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sdict import sdict
from simmanager import SimManager

from lib import Timer, get_random_name, RNNType
from lib.delay_copy_task import DelayCopyData
from lib.models import RNNStatefulWrapper, RNNReadoutWrapper


def train_batch(model, loss_function, optimizer, inputs, targets, masks, train_params, c, device):
    rnn_out, states = model(inputs)

    c_vals = rnn_out
    mean_activity = torch.mean(rnn_out)

    relevant_outputs = model.hidden2out(rnn_out)
    relevant_outputs = torch.masked_select(relevant_outputs, masks).reshape(c.batch_size, c.length, c.width).to(
        device)

    loss = loss_function(relevant_outputs, targets).to(device)

    total_loss = loss
    if c.activity_regularization:  # and it > 200:
        activity_reg_loss = (mean_activity - c.activity_regularization_target) ** 2
        total_loss = total_loss + c.activity_regularization_constant * activity_reg_loss
    if c.voltage_regularization:
        # print("Doing voltage regularization")
        voltage_reg_loss = torch.mean((c_vals - c.voltage_regularization_target) ** 2)
        total_loss = total_loss + c.voltage_regularization_constant * voltage_reg_loss

    optimizer.zero_grad()
    total_loss.backward()

    total_norm = None
    if c.use_grad_clipping:
        total_norm = torch.nn.utils.clip_grad_norm_(train_params, c.grad_clip_norm)

    optimizer.step()

    if c.binary_encoding:
        actual_output = torch.where(torch.sigmoid(relevant_outputs) < 0.5, torch.zeros_like(relevant_outputs),
                                    torch.ones_like(relevant_outputs))
        bitwise_success_rate = (actual_output == targets).float().sum() / (torch.numel(targets))
    else:
        actual_output = torch.argmax(relevant_outputs, -1)
        bitwise_success_rate = (actual_output == torch.argmax(targets, -1)).float().sum() / (
            torch.numel(actual_output))

    return loss, bitwise_success_rate, mean_activity, total_norm, c_vals


def eval_batch(model, loss_function, inputs, targets, masks, c, device):
    rnn_out, states = model(inputs)

    mean_activity_ = torch.mean(rnn_out)

    relevant_outputs = model.hidden2out(rnn_out)
    relevant_outputs = torch.masked_select(relevant_outputs, masks).reshape(c.batch_size, c.length, c.width).to(
        device)

    loss_ = loss_function(relevant_outputs, targets).to(device)

    if c.binary_encoding:
        actual_output = torch.where(torch.sigmoid(relevant_outputs) < 0.5, torch.zeros_like(relevant_outputs),
                                    torch.ones_like(relevant_outputs))
        bitwise_success_rate = (actual_output == targets).float().sum() / (torch.numel(targets))
    else:
        actual_output = torch.argmax(relevant_outputs, -1)
        bitwise_success_rate = (actual_output == torch.argmax(targets, -1)).float().sum() / (
            torch.numel(actual_output))

    return loss_, bitwise_success_rate, mean_activity_


def main(c):
    print('Seed: ', c.seed)

    torch.manual_seed(c.seed)
    np.random.seed(c.seed)

    dataset = DelayCopyData(c.seed, c.length, c.width, c.initial_delay, c.initial_delay_fixed_length, c.delay,
                            c.delay_fixed_length, c.batch_size, c.binary_encoding, blank_symbol=(not c.no_blank_symbol))
    loader = DataLoader(dataset, batch_size=c.batch_size)
    data_iter = iter(loader)

    device = torch.device("cpu")
    if c.cuda:
        device = torch.device("cuda:0")

    input_size = c.total_input_width
    if c.rnn_type == RNNType.LSTM:
        rnn = nn.LSTM(input_size, c.n_units, batch_first=True)
    elif c.rnn_type == RNNType.GRU:
        rnn = nn.GRU(input_size, c.n_units, batch_first=True)
    else:
        raise RuntimeError("Unknown lstm type: %s" % c.rnn_type)
    print("LSTM parameters: ", list(map(lambda x: x[0], rnn.named_parameters())))
    rnn = rnn.to(device)
    # rnn_module = RNNStatefulWrapper(lstm)
    # rnn_module = rnn_module.to(device)

    model = RNNReadoutWrapper(rnn, output_size=c.target_width)
    # del rnn_module, as_module
    model = model.to(device)

    # train_params = list(model.parameters()) + list(as_module.parameters())
    train_params = list(model.parameters())

    if c.use_rmsprop:
        print('Using RMSprop.')
        optimizer = optim.RMSprop(train_params, lr=c.learning_rate, weight_decay=0.9)
    else:
        optimizer = optim.Adam(train_params, lr=c.learning_rate)
    # optimizer = optimizer.to(device)

    if c.binary_encoding:
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    running_avg_bitwise_success_rate = 0.
    model.train()
    for it in range(c.n_training_iterations):

        data = next(data_iter)
        inputs = torch.from_numpy(np.asarray(data['x'])).to(device)
        targets = torch.from_numpy(np.asarray(data['y'])).to(device)
        masks = torch.from_numpy(np.asarray(data['mask'])).to(device)

        # print(inputs.size(), targets.size())

        ep_out_vals = torch.zeros(c.batch_size, c.total_input_length, c.target_width).to(device)
        out_vals = torch.zeros(c.batch_size, c.total_input_length, c.n_units).to(device)

        output_gate_vals = torch.zeros(c.batch_size, c.total_input_length, c.n_units).to(device)

        ## TRAINING
        with Timer() as bt:
            loss, bitwise_success_rate, mean_activity, total_norm, c_vals = train_batch(model, loss_function, optimizer,
                                                                                        inputs, targets, masks,
                                                                                        train_params, c,
                                                                                        device)

            running_avg_bitwise_success_rate += bitwise_success_rate.data.item()
            running_avg_bitwise_success_rate /= 2
            if running_avg_bitwise_success_rate > 0.98:
                print(
                    f"Training iteration {it} :: Loss is {loss.data.item():.4f} :: Running avg. of bitwise success rate is high enough {running_avg_bitwise_success_rate:.4f}. Stopping training.")
                break

        if it % 100 == 0 or it == 1 or it == c.n_training_iterations - 1:
            print(f"Training iteration {it} :: Loss is {loss.data.item():.4f} :: Bitwise success rate"
                  f" {bitwise_success_rate.data.item():.4f} (Running avg.  {running_avg_bitwise_success_rate:.4f}) ::"
                  f" Mean activity {mean_activity.data.item():.4f} :: "
                  f" Batch time was {bt.difftime:.4f}.")
            if c.use_grad_clipping:
                print(f'Total norm of gradients before clipping: {total_norm:.4f}')

    ## TESTING
    model.eval()
    with torch.no_grad():
        for it in range(c.n_testing_iterations):
            data = next(data_iter)
            inputs = torch.from_numpy(np.asarray(data['x'])).to(device)
            targets = torch.from_numpy(np.asarray(data['y'])).to(device)
            masks = torch.from_numpy(np.asarray(data['mask'])).to(device)

            loss, bitwise_success_rate, mean_activity = eval_batch(model, loss_function, inputs, targets, masks, c,
                                                                   device)

            if it % 100 == 0 or it == 1 or it == c.n_testing_iterations - 1:
                print(f"Testing iteration {it} :: Loss is {loss.data.item():.4f} :: "
                      # f" Activity sparsity {activity_sparsity.data.item():.4f}"
                      f" Mean activity {mean_activity.data.item():.4f} :: "
                      f"Bitwise success rate {bitwise_success_rate.data.item():.4f} :: Batch time was {bt.difftime:.4f}.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=3000)
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--learning-rate', type=float, default=0.001)
    argparser.add_argument('--use-rmsprop', action='store_true')
    argparser.add_argument('--use-grad-clipping', action='store_true')
    argparser.add_argument('--grad-clip-norm', type=float, default=2.0)
    argparser.add_argument('--initial-delay', type=int, required=True)  # Delay before string presentation.
    argparser.add_argument('--initial-delay-variable-length', action='store_true')  # Delay before string presentation.
    argparser.add_argument('--delay', type=int, required=True)  # Delay after string presentation.
    argparser.add_argument('--delay-variable-length', action='store_true')  # Delay after string presentation.
    argparser.add_argument('--no-blank-symbol', action='store_true')
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--rnn-type', type=str, default='lstm', choices=[e.value for e in RNNType])
    argparser.add_argument('--train-iter', type=int, default=10000)
    argparser.add_argument('--activity-regularization', action='store_true')
    argparser.add_argument('--activity-regularization-constant', type=float, default=1.)
    argparser.add_argument('--activity-regularization-target', type=float, default=0.05)
    argparser.add_argument('--debug', action='store_true')
    args = argparser.parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)


    # START CONFIG
    def get_config():
        print('Generating dictionary of parameters')
        # General
        seed = args.seed

        # Task specific parameters
        initial_delay = args.initial_delay
        initial_delay_fixed_length = not args.initial_delay_variable_length
        delay = args.delay
        delay_fixed_length = not args.delay_variable_length
        length = 2
        binary_encoding = True  # False # True
        width = 3  # 8 # in bits if binary encoding, number of symbols otherwise, for example 8

        # (LSTM) Network parameters

        n_training_iterations = args.train_iter
        n_testing_iterations = 100

        # Derived parameters
        if args.no_blank_symbol:
            total_input_width = width + 1
        else:
            total_input_width = width + 2
        total_input_length = 2 * length + initial_delay + delay
        target_length = length
        target_width = width

        # Convert string argument to enum
        for e in RNNType:
            if args.rnn_type == e.value:
                rnn_type = e
                break
        else:
            raise RuntimeError(f"Unknown value {args.rnn_type}")

        if rnn_type in [RNNType.LSTM, RNNType.GRU]:
            n_units = 50  # 256
        else:
            raise RuntimeError(f"Unknown RNN type {rnn_type}")

        if args.activity_regularization:
            print("using activity regularization")

        if args.debug:
            print("!!DEBUG!!")
            n_training_iterations = 10
            n_testing_iterations = 10

        config = dict(
            n_training_iterations=n_training_iterations,
            n_testing_iterations=n_testing_iterations,
            seed=seed,
            cuda=args.cuda,
            rnn_type=rnn_type,
            length=length,
            width=width,
            initial_delay=initial_delay,
            initial_delay_fixed_length=initial_delay_fixed_length,
            delay=delay,
            delay_fixed_length=delay_fixed_length,
            no_blank_symbol=args.no_blank_symbol,
            batch_size=args.batch_size,
            n_units=n_units,
            learning_rate=args.learning_rate,
            use_rmsprop=args.use_rmsprop,
            use_grad_clipping=args.use_grad_clipping,
            grad_clip_norm=args.grad_clip_norm,
            binary_encoding=binary_encoding,
            total_input_width=total_input_width,
            total_input_length=total_input_length,
            target_length=target_length,
            target_width=target_width,
            activity_regularization=args.activity_regularization,
            activity_regularization_constant=args.activity_regularization_constant,
            activity_regularization_target=args.activity_regularization_target,
        )
        print(config)
        config = sdict(config)
        return config


    ## END CONFIG
    config = get_config()

    ## START DIR NAMES
    rroot = os.path.expanduser(os.path.join('~', 'output'))
    ## END DIR NAMES

    ## START DIR NAMES
    root_dir = os.path.join(rroot, 'es-rnn')
    if args.debug:
        root_dir = os.path.expanduser(os.path.join(rroot, 'tmp'))  # NOTE: DEBUG
    os.makedirs(root_dir, exist_ok=True)
    sim_name = get_random_name()
    ## END DIR NAMES

    with SimManager(sim_name, root_dir, write_protect_dirs=False, tee_stdx_to='output.log') as simman:
        paths = simman.paths
        print("Results will be stored in ", paths.results_path)

        with open(os.path.join(paths.data_path, 'config.yaml'), 'w') as f:
            yaml.dump(config.todict(), f, allow_unicode=True, default_flow_style=False)

        print('Calling main')
        if args.debug:
            from ipdb import launch_ipdb_on_exception

            with launch_ipdb_on_exception():
                main(config)
        else:
            main(config)

        # make_plots(paths.results_path, config.total_input_width)
        print("Results stored in ", paths.results_path)
