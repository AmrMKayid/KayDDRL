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

