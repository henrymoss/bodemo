import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from trieste.objectives import Branin
from trieste.objectives.utils import mk_observer
from trieste.models.gpflow.builders import build_gpr
from trieste.models.gpflow import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.function import ExpectedImprovement
from trieste.acquisition.function import expected_improvement
from trieste.data import Dataset
from trieste.bayesian_optimizer import BayesianOptimizer
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import cast



class ConfigurableAcq(ExpectedImprovement):
	def __init__(self, acq, search_space= None):
		self._search_space = search_space
		self._acq = acq
	def prepare_acquisition_function(self,model, dataset):
		tf.debugging.Assert(dataset is not None, [tf.constant([])])
		dataset = cast(Dataset, dataset)
		tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

		# Check feasibility against any explicit constraints in the search space.
		if self._search_space is not None and self._search_space.has_constraints:
			is_feasible = self._search_space.is_feasible(dataset.query_points)
			if not tf.reduce_any(is_feasible):
				query_points = dataset.query_points
			else:
				query_points = tf.boolean_mask(dataset.query_points, is_feasible)
		else:
			is_feasible = tf.constant([True], dtype=bool)
			query_points = dataset.query_points

		mean, _ = model.predict(query_points)
		if not tf.reduce_any(is_feasible):
			eta = tf.reduce_max(mean, axis=0)
		else:
			eta = tf.reduce_min(mean, axis=0)

		return self._acq(model, eta)


def run_bo(acquisition_fn = expected_improvement, problem = Branin, num_steps =10, num_reps=1, seed=1234):

	observer = mk_observer(problem.objective)
	search_space = problem.search_space
	minimum = problem.minimum
	regrets = []
	acquisition = ConfigurableAcq(acquisition_fn)

	for i in range(num_reps):
		print(f"Performing rep {i} of {num_reps}")
		tf.random.set_seed(i*seed)
		initial_query_points = search_space.sample(2 * search_space.dimension + 1)
		initial_data = observer(initial_query_points)
		gpflow_model = build_gpr(initial_data, search_space)
		model = GaussianProcessRegression(gpflow_model, num_kernel_samples=0)
		acquisition_rule = EfficientGlobalOptimization(
			acquisition,
			optimizer = generate_continuous_optimizer()
			)
		bo = BayesianOptimizer(observer, search_space, )
		result = bo.optimize(num_steps, initial_data, model, acquisition_rule=acquisition_rule)
		regret = np.minimum.accumulate(result.try_get_final_dataset().observations - minimum)[:,0]
		regrets.append(regret)


	return tf.math.log(tf.stack(regrets,-1))





if __name__ == '__main__':
	run_bo(ConfigurableAcq(custom_acquisition_function))

	