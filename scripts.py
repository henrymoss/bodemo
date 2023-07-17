import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from trieste.objectives import Branin
from trieste.objectives.utils import mk_observer
from trieste.models.gpflow.builders import build_gpr
from trieste.models.gpflow import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedImprovement
from trieste.bayesian_optimizer import BayesianOptimizer
import matplotlib.pyplot as plt
import numpy as np



def run_and_plot(acquisition_fn = ExpectedImprovement()):
	problem = Branin
	num_steps =10
	noise_var = 1e-6
	num_reps=10

	new_regrets = run_bo(acquisition_fn, problem, num_steps, noise_var, num_reps):

	plt.figure()
	plt.xlabel("# Optimisation Steps")
	plt.ylabel("Regret")
	for i in range(tf.shape(regrets)[1]):
		plt.plot(regrets[:,i])


def run_bo(acquisition_fn = ExpectedImprovement(), problem = Branin, num_steps =10, noise_var = 1e-6, num_reps=10):

	observer = mk_observer(problem.objective)
	search_space = problem.search_space
	minimum = problem.minimum
	regrets = []

	for i in range(num_reps):
		print(f"Performing rep {i} of {num_reps}")
		initial_query_points = search_space.sample(2 * search_space.dimension + 2)
		initial_data = observer(initial_query_points)
		gpflow_model = build_gpr(initial_data, search_space)
		model = GaussianProcessRegression(gpflow_model)
		acquisition_rule = EfficientGlobalOptimization(
			acquisition_fn,
			optimizer = generate_continous_optimizer()
			)
		bo = BayesianOptimizer(observer, search_space)
		result = bo.optimize(num_steps, initial_data, model)
		regret = np.minimum.accumulate(result.try_get_final_dataset().observations - minimum)
		regrets.append(regret)


	return tf.stack(regrets,-1)





if __name__ == '__main__':
	run_bo()

    