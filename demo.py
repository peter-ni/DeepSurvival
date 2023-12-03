import jax
import jax.numpy as jnp


def sum_of_squares(x):
    return jnp.sum(x**2)

def squared(x):
    return 

x = jnp.array([1.0, 2.0, 3.0, 4.0])


squared_dx = jax.grad(squared)



# sum_of_squares_dx = jax.grad(sum_of_squares)
# print(sum_of_squares(x))
# print(sum_of_squares_dx(x))




