import jax
import jax.numpy as jnp

from evosax import OpenES, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
# from evosax.problems import GymFitness

from DelaySequence import *

seed = 3000
length = 2
width = 3
initial_delay = 0
initial_delay_fixed_length = True
delay = 2
delay_fixed_length = True
batch_size = 1000
binary_encoding = True
blank_symbol = True

learning_rate = 0.001
steps = 10000
hidden_size = 50

insize = width + 1
outsize = width

if blank_symbol:
    insize = width + 2

rng = jax.random.PRNGKey(0)

#
network = NetworkMapper["LSTM"]( 
        num_hidden_units = 50,
        num_output_units = 3,
        output_activation = "categorical"
    )


pholder = jnp.zeros((insize,))
carry_init = network.initialize_carry()

params = network.init(
    rng,
    x=pholder,
    carry=carry_init,
    rng=rng,
)

param_reshaper = ParameterReshaper(params)

evaluator = DelaySequenceFitness(batch_size = 100)
evaluator.set_apply_fn(param_reshaper.vmap_dict, network.apply, network.initialize_carry)



popsize = 100
strategy = PGPE(param_reshaper.total_params, popsize,
                elite_ratio=0.1, opt_name="adam")

# Update basic parameters of PGPE strategy
es_params = strategy.default_params.replace(
        sigma_init=0.05,  # Initial scale of isotropic Gaussian noise
        sigma_decay=0.999,  # Multiplicative decay factor
        sigma_limit=0.01,  # Smallest possible scale
        sigma_lrate=0.2,  # Learning rate for scale
        sigma_max_change=0.2,  # clips adaptive sigma to 20%
        init_min=-0.1,  # Range of parameter mean initialization - Min
        init_max=0.1,  # Range of parameter mean initialization - Max
        clip_min=-10,  # Range of parameter proposals - Min
        clip_max=10  # Range of parameter proposals - Max
)

# Update optimizer-specific parameters of Adam
es_params = es_params.replace(opt_params=es_params.opt_params.replace(
        lrate_init=0.05,  # Initial learning rate
        lrate_decay=0.999, # Multiplicative decay factor
        lrate_limit=0.01,  # Smallest possible lrate
        beta_1=0.99,   # Adam - beta_1
        beta_2=0.999,  # Adam - beta_2
        eps=1e-8,  # eps constant,
        )
)

print(es_params)

num_generations = 200
print_every_k_gens = 20

es_logging = ESLog(param_reshaper.total_params,
                   num_generations,
                   top_k=5,
                   maximize=True)
log = es_logging.initialize()

fit_shaper = FitnessShaper(w_decay=0.1,
                           maximize=True)

state = strategy.initialize(rng, es_params)

for gen in range(num_generations):
    rng, rng_init, rng_ask, rng_eval = jax.random.split(rng, 4)
    x, state = strategy.ask(rng_ask, state, es_params)
    reshaped_params = param_reshaper.reshape(x)

    fitness = evaluator.rollout(rng_eval, reshaped_params).mean(axis=1) 
    fit_re = fit_shaper.apply(x, fitness)
    state = strategy.tell(x, fit_re, state, es_params)
    log = es_logging.update(log, x, fitness)
    
    if gen % print_every_k_gens == 0:
        print("Generation: ", gen, "Performance: ", log["log_top_1"][gen])