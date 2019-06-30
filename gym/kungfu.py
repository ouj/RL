import retro

env = retro.make(game='KungFu-Nes')

def play_once(env, render=False):
    observation = env.reset()
    done = False
    steps = 0
    total_return = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        steps += 1
        total_return += reward
        if render:
            env.render()
    return steps, total_return

play_once(env, render=True)

env.close()
