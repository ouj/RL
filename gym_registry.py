from gym import envs
import pprint
for e in list(envs.registry.all()):
    pprint.pprint(e.id)
