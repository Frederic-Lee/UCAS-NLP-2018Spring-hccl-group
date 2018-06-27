import time
import copy

class Dict2Obj(object):
    def __init__(self, map, recur=False):
        self.map = map
        self.recur = recur

    def __setattr__(self, name, value):
        if name == 'map':
            object.__setattr__(self, name, value)
            return;
        self.map[name] = value

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return self.map[name];
        """
        v = self.map[name]
        if self.recur is True: 
            if isinstance(v, (dict)):
                return Dict2Obj(v)
            if isinstance(v, (list)):
                r = []
                for i in v:
                    r.append(Dict2Obj(i))
                return r  
            else:
                return self.map[name];
        else:
            return self.map[name];
        """

    def __getitem__(self, name):
        return self.map[name]
