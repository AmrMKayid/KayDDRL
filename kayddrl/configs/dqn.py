config = {
    "engine": {
        "num_episodes": 2000,
        "to_be_solved_score": 100000.0,
        "log_dir": "runs/",
    },
    "env": {
        # "name": "Reacher-v2",
        "name": "CartPole-v1",
        "type": "openai",
        "seed": 0,
        "to_render": True,
        "frame_sleep": 0.001,
        "max_steps": 1000,
        "one_hot": 1,
        "action_bins": 1,
        "reward_scale": 1,
        "num_envs": 1,
    },
    "agent": {
        "name": "DQNAgent",
        "algo": {
            "name": "DQN",
            # "action_prob_dist_type": "MultivariateNormal",
            "action_prob_dist_type": "Argmax",
            "action_policy": "epsilon_greedy",
            "explore_var_config": {
                "name": "linear_decay",
                "start_val": 1.0,
                "end_val": 0.1,
                "start_step": 0,
                "end_step": 1000
            },
            "gamma": 0.99,
            "lr": 5e-4,
            "update_every": 4,
        },
        "memory": {
            'name': 'ReplayBuffer',
            'buffer_size': 10000,
            'batch_size': 32,
            'device': 'cpu',
        },
        "net": {
            "name": "NeuralNet",
            "type": "MLP",
            "hid_layers": [
                64, 32, 16
            ],
            "hid_layers_activation": "selu",
            "clip_grad_val": 0.5,
            "loss_config": {
                "name": "MSELoss"
            },
            "optim_config": {
                "name": "Adam",
                "lr": 0.02
            },
            "lr_scheduler_config": {
                "name": "StepLR",
                "step_size": 1000,
                "gamma": 0.9
            },
            "update_type": "polyak",
            "update_frequency": 32,
            "tau_polyak_coef": 0.1,
            "gpu": True
        }
    },
}
