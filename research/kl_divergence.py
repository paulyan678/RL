import numpy as np
# generate 2d grid of ramdom interger value
np.random.seed(0)
a = np.random.randint(1, 10, (10, 10))
b = np.random.randint(1, 10, (10, 10))
a = a / a.sum()
b = b / b.sum()

# calculate kl divergence
# b is what we think we're sampling from. a is what we're actually sampling from
kl_div_b_think_a_act = (a * np.log(a / b)).sum()
print(kl_div_b_think_a_act)

kl_div_a_think_b_act = (b * np.log(b / a)).sum()
print(kl_div_a_think_b_act)
