'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.AC import ActorCriticAgent
from agent.hand import HandAgent
from simulator import ICRABattleField
from utils import Action, ID_R1, ID_B1

parser = argparse.ArgumentParser(
    description="Train the model in the ICRA 2019 Battlefield")
parser.add_argument("--seed", type=int, default=233, help="Random seed")
parser.add_argument("--enemy", type=str, default="hand",
                    help="The opposite agent type [AC, hand]")
parser.add_argument("--load_model", action='store_true',
                    help="Whether to load the trained model")
parser.add_argument("--load_model_path", type=str,
                    default="ICRA.model", help="The path of trained model")
parser.add_argument("--save_model_path", type=str,
                    default="ICRA_save.model", help="The path of trained model")
parser.add_argument("--epoch", type=int, default=1000,
                    help="Number of epoches to train")
parser.add_argument("--update_step", type=int, default=10,
                    help="After how many step, update the model?")
args = parser.parse_args()


torch.random.manual_seed(args.seed)
torch.cuda.random.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

agent = ActorCriticAgent()
if args.load_model:
    agent.load_model(args.load_model_path)
if args.enemy == "hand":
    agent2 = HandAgent()
elif args.enemy == "AC":
    agent2 = ActorCriticAgent()
    agent2.load_model(args.load_model_path)

env = ICRABattleField()
env.seed(args.seed)
losses = []
rewards = []
for i_episode in range(1, args.epoch + 1):
    print("Epoch: [{}/{}]".format(i_episode, args.epoch))
    # Initialize the environment and state
    action = Action()
    pos = env.reset()
    if args.enemy == "hand":
        agent2.reset(pos)
    state, reward, done, info = env.step(action)
    for t in (range(2*60*30)):
        # Other agent
        if args.enemy == "hand":
            env.set_robot_action(ID_B1, agent2.select_action(state[ID_B1]))
        elif args.enemy == "AC":
            env.set_robot_action(ID_B1, agent2.select_action(
                state[ID_B1], mode="max_probability"))

        # Select and perform an action
        state_map = agent.preprocess(state[ID_R1])
        a_m, a_t = agent.run_AC(state_map)
        action = agent.decode_action(a_m, a_t, state[ID_R1], "max_probability")

        # Step
        next_state, reward, done, info = env.step(action)
        tensor_next_state = agent.preprocess(next_state[ID_R1])

        # Store the transition in memory
        agent.push(state_map, tensor_next_state, [a_m, a_t], [reward])
        state = next_state
        state_map = tensor_next_state

        # env.render()
        # Perform one step of the optimization (on the target network)
        if done:
            break

    print("Simulation end in: {}:{:02d}, reward: {}".format(
        t//(60*30), t % (60*30)//30, env.reward))
    agent.memory.finish_epoch()
    loss = agent.optimize_offline(1)
    losses.append(loss)
    rewards.append(env.reward)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % args.update_step == 0:
        agent.update_target_net()
        agent.save_model(args.save_model_path)

print('Complete')
env.close()

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.savefig("loss.pdf")

plt.title("Reward")
plt.xlabel("Epoch")
plt.ylabel("Final reward")
plt.plot(rewards)
plt.savefig("reward.pdf")
