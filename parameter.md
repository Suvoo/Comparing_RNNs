{
'n_training_iterations': 10000, 
'n_testing_iterations': 100, 
'seed': 3000, 
'cuda': False, 
'rnn_type': <RNNType.GRU: 'gru'>, 
'length': 2, 
'width': 3, 
'initial_delay': 1, 
'initial_delay_fixed_length': True, 
'delay': 2, 
'delay_fixed_length': True, 
'no_blank_symbol': False, 
'batch_size': 1000, 
'n_units': 50, 
'learning_rate': 0.001, 
'use_rmsprop': False, 
'use_grad_clipping': False, 
'grad_clip_norm': 2.0, 
'binary_encoding': True, 
'total_input_width': 5, 
'total_input_length': 7, 
'target_length': 2, 
'target_width': 3, 
'activity_regularization': False, 
'activity_regularization_constant': 1.0, 
'activity_regularization_target': 0.05
}

# Results 

Seed:  3000
LSTM parameters:  ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
Training iteration 0 :: Loss is 0.7040 :: Bitwise success rate 0.4290 (Running avg.  0.2145) :: Mean activity 0.0041 ::  Batch time was 0.1784.

Training iteration 1 :: Loss is 0.7021 :: Bitwise success rate 0.4308 (Running avg.  0.3227) :: Mean activity 0.0033 ::  Batch time was 0.1632.

Training iteration 100 :: Loss is 0.4692 :: Bitwise success rate 0.7607 (Running avg.  0.7597) :: Mean activity -0.0215 ::  Batch time was 0.3571.

Training iteration 200 :: Loss is 0.3065 :: Bitwise success rate 0.8523 (Running avg.  0.8496) :: Mean activity -0.0230 ::  Batch time was 0.1546.

Training iteration 273 :: Loss is 0.1485 :: Running avg. of bitwise success rate is high enough 0.9802. Stopping training.


Testing iteration 0 :: Loss is 0.1436 ::  Mean activity -0.0390 :: Bitwise success rate 0.9815 :: Batch time was 0.1237.
Testing iteration 1 :: Loss is 0.1470 ::  Mean activity -0.0396 :: Bitwise success rate 0.9760 :: Batch time was 0.1237.
Testing iteration 99 :: Loss is 0.1438 ::  Mean activity -0.0393 :: Bitwise success rate 0.9802 :: Batch time was 0.1237.
