from kayddrl.configs.default import default
from kayddrl.demo.ddpg_test import run
from kayddrl.envs import make_env
from kayddrl.utils.logging import logger
from kayddrl.utils.utils import set_global_seeds

if __name__ == '__main__':
    cfg = default()
    set_global_seeds(cfg.glob.seed)
    env = make_env(cfg.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logger.info(env.action_space.low, env.action_space.high, env.observation_space.shape[0])
    run(env, cfg, state_dim, action_dim)