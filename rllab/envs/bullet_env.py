from rllab.envs.gym_env import GymEnv
import pybullet_envs
import pybullet as p
import inspect

class BulletEnv(GymEnv):
    def __init__(self,
                 env_name,
                 record_video=True,
                 video_schedule=None,
                 log_dir=None,
                 record_log=True,
                 force_reset=False):
        
        super(BulletEnv, self).__init__(env_name, record_video, video_schedule, log_dir, record_log, force_reset)

    def patch_deprecated_methods(self, env):

        env.reset = env._reset
        env.step  = env._step
        env.seed  = env._seed
        def render(mode):
            return env._render(mode, close=False)
        def close():
            env._render("human", close=True)
        env.render = render
        env.close = close

        return env

    def render(self):

        env = self.env.env.env
        env_cls = env.__class__

        if not env._renders:
            try:
                p.disconnect()
                tmp_env = env_cls(renders=True)
                # p.setTimeStep(10)
                p.setRealTimeSimulation(0)
                tmp_env = self.patch_deprecated_methods(tmp_env)
                tmp_env.__dict__["_observation"] = env.__dict__["_observation"]
                self.env.env.env = tmp_env
            except:
                pass

        self.env.render(mode='rgb_array')