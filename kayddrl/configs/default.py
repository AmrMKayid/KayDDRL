import torch


class Configs(object):
    def __init__(self, **kwargs):
        self.__dict__ = dict(kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def pprint(self, d, indent=0):
        for key, value in d.items():
            if isinstance(value, Configs):
                print("%s = {" % key)
                self.pprint(value, indent=1)
                print("}")
                print()
            else:
                print("\t" * indent + "{} = {}".format(key, value))

    def __repr__(self):
        return repr(self.pprint(self.__dict__))

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()


def default():
    return Configs(
        glob=Configs(
            seed=7,
            test=False,
            load_from=None,
            interim_test_num=200,
            name="DDPG_Experiment",
            max_episode_steps=-1,
            save_period=200,
            num_episodes=20000,
            interim_test_nums=10,
            log=True,
            log_dir="runs/",
            global_step=0,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        ),
        env=Configs(
            # name="Reacher-v2",
            # type="openai",
            name="Reacher",
            type="unity",
            seed=777,
            to_render=False,
            frame_sleep=0.001,
            max_steps=1000,
            one_hot=1,
            action_bins=1,
            reward_scale=1,
            num_envs=1,
        ),
        agent=Configs(
            name="DDPG",
            gamma=0.99,
            tau=1e-3,
            multiple_learn=1,
            gradient_clip_cr=1.0,
            gradient_clip_ac=0.5,
            initial_random_action=10000,
            ou_noise_theta=0.0,
            ou_noise_sigma=0.0,
        ),
        models=Configs(
            lr_actor=1e-3,
            lr_critic=1e-3,
            weight_decay=1e-6,
        ),
        memory=Configs(
            name="ReplyBuffer",
            buffer_size=1000,
            batch_size=128,
        )
    )
