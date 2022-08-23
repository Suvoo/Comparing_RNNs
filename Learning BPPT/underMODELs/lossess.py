from xml.etree.ElementPath import prepare_descendant
import torch

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

import optax

def BCEWithLLogitLLoss():
    target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
    # print(target.shape)
    output = torch.full([10, 64], 1.5)  # A prediction (logit)
    # print(output)
    pos_weight = torch.ones([64])  # All weights are equal to 1
    # print(pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(criterion)
    ans = criterion(output, target)  # -log(sigmoid(1.5))
    print(ans)

    print('=====================================================================')

    target = jnp.ones((10,64))
    # print(target.shape)
    output = jnp.full((10,64),1.5)
    # print(output)
    pos_weight = jnp.ones(64,)
    # print(pos_weight)
    # loss_fn = optax.sigmoid_binary_cross_entropy
    criterion = optax.sigmoid_binary_cross_entropy(output,target)
    ans = jnp.mean(criterion)
    print(ans)

    print('=====================================================================')
    y,pred_y = target,output
    
    pred_y = jax.nn.sigmoid(pred_y)
    ans2 = -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
    print(ans2)

def CrossEEntropyLLoss():
    target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
    # print(target.shape)
    output = torch.full([10, 64], 1.5)  # A prediction (logit)
    # print(output)
    pos_weight = torch.ones([64])  # All weights are equal to 1
    # print(pos_weight)
    criterion = torch.nn.CrossEntropyLoss()
    print(criterion)
    ans = criterion(output, target)  # -log(sigmoid(1.5))
    print(ans)

    print('=====================================================================')

    target = jnp.ones((10,64))
    # print(target.shape)
    output = jnp.full((10,64),1.5)
    # print(output)
    pos_weight = jnp.ones(64,)
    # print(pos_weight)
    criterion = optax.softmax_cross_entropy(output,target)
    ans = -jnp.mean(criterion)
    print(ans)



    
BCEWithLLogitLLoss()
# CrossEEntropyLLoss()

# # def CrossEEntropyLLoss():
# # Example of target with class indices
# criteria = torch.nn.CrossEntropyLoss()
# input = torch.tensor([[3.4, 1.5,0.4, 0.10]],dtype=torch.float)
# target = torch.tensor([0], dtype=torch.long)
# print(input.shape,target.shape)
# print(criteria(input, target))

# input = jnp.array([[3.4, 1.5,0.4, 0.10]],dtype = jnp.float32)
# target = jnp.array([0], dtype=jnp.float32)
# criteria = optax.softmax_cross_entropy(input,target)
# print(input.shape,target.shape)
# print(criteria)
