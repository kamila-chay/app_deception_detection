from scipy import stats
import numpy as np

values = [1.0] * 55 + [0.95, 0.85, 0.75, 0.92, 0.97, 0.67, 0.75, 0.95]

data = np.array(values)  

t_statistic, p_value_two_sided = stats.ttest_1samp(data, 0.96)

# For a one-sided test (H₁: mean > 0.95)
p_value_one_sided = p_value_two_sided / 2 if t_statistic > 0 else 1 - p_value_two_sided / 2

print("t-statistic:", t_statistic)
print("one-sided p-value:", p_value_one_sided)