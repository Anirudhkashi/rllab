from gym.envs.registration import register
import datetime

class Register():
	# Registers new env into gym.

	def __init__(self, folder, env_class_name, max_episode_steps=1000, reward_threshold=25.0, params=None):

		'''
        Args:
        folder(String): relative path where the .py file containing the class is present.
        env_object(String): Name of the class which inherits gym.Env
        max_episode_steps: The gym env parameter
        reward_threshold: The gym env parameter
        params(dict): Optional key value parameters to be passed to the class.
        '''

		now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

		self._entry_point = folder + ":" + env_class_name
		self._id = env_class_name + "-v" + now
		self._max_episode_steps = max_episode_steps
		self._reward_threshold = reward_threshold
		self._params = params


	def register_env(self):

		if self._params is None:
			register(
					id = self._id,
					entry_point = self._entry_point,
		            max_episode_steps = self._max_episode_steps,
		            reward_threshold = self._reward_threshold,
				)
		else:
			register(
					id = self._id,
					entry_point = self._entry_point,
		            kwargs = self._params,
		            max_episode_steps = self._max_episode_steps,
		            reward_threshold = self._reward_threshold,
				)

	def get_id(self):
		return self._id