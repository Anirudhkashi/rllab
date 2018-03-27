import gym
import gym.wrappers
import gym.envs
import traceback
import logging

try:
    from gym.wrappers.monitoring import logger as monitor_logger

    monitor_logger.setLevel(logging.WARNING)
except Exception as e:
    traceback.print_exc()

import os
import os.path as osp
from rllab.envs.gym_env import convert_gym_space, CappedCubicVideoSchedule, FixedIntervalVideoSchedule, NoVideoSchedule, GymEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger


class BulletEnv(GymEnv, Serializable):
    def __init__(self,
                 env_name,
                 record_video=True,
                 video_schedule=None,
                 log_dir=None,
                 record_log=True,
                 force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log(
                    "Warning: skipping Bullet environment monitoring since snapshot_dir not configured."
                )
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "bullet_log")
        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)
        self.env = env
        self.env_id = env.spec.id

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(
                self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.spec.max_episode_steps
        self._log_dir = log_dir
        self._force_reset = force_reset
