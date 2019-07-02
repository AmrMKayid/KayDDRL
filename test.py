from kayddrl.envs import make_env
from kayddrl.configs.dqn import config
from kayddrl.agents.base import BaseAgent
from kayddrl.engine.engine import Engine

# if __name__ == '__main__':
#     env = make_env(config["env"])
#     agent = BaseAgent(env, config["agent"])
#
#     engine = Engine(env, agent, config["engine"])
#     engine.train()
from kayddrl.utils.logging import logger

if __name__ == '__main__':
    #
    #
    #
    #
    #     # print("# -------------------- Memory -------------------- #")
    #     # mem_config = {
    #     #     'name': 'ReplayBuffer',
    #     #     'buffer_size': 10000,
    #     #     'batch_size': 32,
    #     #     'device': 'cpu',
    #     # }
    #     # memory = ReplayBuffer(mem_config)
    #     # memory.update(state=1, action=0, reward=0, next_state=2, done=0)
    #     # memory.update(state=2, action=1, reward=1, next_state=3, done=0)
    #     # memory.update(state=3, action=0, reward=0, next_state=4, done=0)
    #     # memory.update(state=4, action=1, reward=-1, next_state=5, done=0)
    #     # memory.update(state=5, action=0, reward=0, next_state=6, done=1)
    #     # states, actions, rewards, next_states, dones = memory.sample()
    #     # print(states, actions, rewards, next_states, dones)
    #
    print("# -------------------- Envs -------------------- #")
    env_config = {
        "name": "Reacher-v2",
        # "name": "Reacher-v2",
        "type": "openai",
        "seed": 0,
        "to_render": False,
        "frame_sleep": 0.001,
        "max_steps": 1000,
        "one_hot": 1,
        "action_bins": 1,
        "reward_scale": 1,
        "num_envs": 1,
    }
    env = make_env(env_config)
    logger.info(env.action_space.low, env.action_space.high, env.observation_space.shape[0])
    # #
    # for i in range(3):
    #     observation = env.reset()
    #     for j in range(100):
    #         state = observation
    #         action = env.action_space.sample()
    #         print(type(action))
    #         observation, reward, done, info = env.step(action)
    #         next_state = observation
    #         print("state : {0}, action : {1}, next_state : {2}, reward : {3}, done : {4}, info : {5}".format(
    #             state, action, next_state, reward, done, info))
    #
    #         if done:
    #             print("You reached your goal!")
    #             break
    # env.close()

    # env_config = {
    #     "name": "Reacher",
    #     "type": "unity",
    #     "seed": 0,
    #     "to_render": True,
    #     "frame_sleep": 0.001,
    #     "max_steps": 1000,
    #     "one_hot": 1,
    #     "action_bins": 1,
    #     "reward_scale": 1,
    #     "num_envs": 1,
    # }
    # env = make_env(env_config)
    #
    # for i in range(37):
    #     observation = env.reset()
    #     for j in range(100):
    #         state = observation
    #         action = env.action_space.sample()
    #         print(type(action))
    #         observation, reward, done, info = env.step(action)
    #         next_state = observation
    #         # print("state : {0}, action : {1}, next_state : {2}, reward : {3}, done : {4}, info : {5}".format(
    #         #     state, action, next_state, reward, done, info))
    #
    #         if done:
    #             print("You reached your goal!")
    #             break
    # env.close()
#
#     print("# -------------------- Nets -------------------- #")
#
#     # net_config = {
#     #     "name": "NeuralNet",
#     #     "type": "MLP",
#     #     "hid_layers": [
#     #         64, 32, 16
#     #     ],
#     #     "hid_layers_activation": "selu",
#     #     "clip_grad_val": 0.5,
#     #     "loss_config": {
#     #         "name": "MSELoss"
#     #     },
#     #     "optim_config": {
#     #         "name": "Adam",
#     #         "lr": 0.02
#     #     },
#     #     "lr_scheduler_config": {
#     #         "name": "StepLR",
#     #         "step_size": 1000,
#     #         "gamma": 0.9
#     #     },
#     #     "update_type": "polyak",
#     #     "update_frequency": 32,
#     #     "polyak_coef": 0.1,
#     #     "gpu": True
#     # }
#     # net = MLP(net_config, 4, 2)
#     # print(net)
#
#     print("# -------------------- Agent -------------------- #")
#     agent_config = {
#         "name": "DQNAgent",
#         "algo": {
#             "name": "DQN",
#             "action_prob_dist_type": "MultivariateNormal",
#             "action_policy": "epsilon_greedy",
#             "explore_var_config": {
#                 "name": "linear_decay",
#                 "start_val": 1.0,
#                 "end_val": 0.1,
#                 "start_step": 0,
#                 "end_step": 1000
#             },
#             "gamma": 0.99,
#             "lr": 5e-4,
#             "update_every": 4,
#             "training_batch_iter": 8,
#             "training_iter": 4,
#             "training_frequency": 4,
#             "training_start_step": 32
#         },
#         "memory": {
#             'name': 'ReplayBuffer',
#             'buffer_size': 10000,
#             'batch_size': 32,
#             'device': 'cpu',
#         },
#         "net": {
#             "name": "NeuralNet",
#             "type": "MLP",
#             "hid_layers": [
#                 64, 32, 16
#             ],
#             "hid_layers_activation": "selu",
#             "clip_grad_val": 0.5,
#             "loss_config": {
#                 "name": "MSELoss"
#             },
#             "optim_config": {
#                 "name": "Adam",
#                 "lr": 0.02
#             },
#             "lr_scheduler_config": {
#                 "name": "StepLR",
#                 "step_size": 1000,
#                 "gamma": 0.9
#             },
#             "update_type": "polyak",
#             "update_frequency": 32,
#             "polyak_coef": 0.1,
#             "gpu": True
#         }
#     }
#     agent = BaseAgent(env, agent_config)
#
#     engine = Engine(env, agent, {
#         'num_episodes': 2000,
#         'max_t': 1000,
#         'to_be_solved_score': 100000.0,
#         'log_dir': 'runs/',
#     })
#     engine.train()
