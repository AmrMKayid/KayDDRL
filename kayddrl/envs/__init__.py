# the environment module


def make_env(config):
    env = None
    env_type = config['type']  # config['env']['type']

    if env_type.lower() == "openai":
        from kayddrl.envs.openai import GymEnv
        env = GymEnv(config)
    elif env_type.lower() == 'unity':
        from kayddrl.envs.unity import UnityEnv
        env = UnityEnv(config)

    return env
