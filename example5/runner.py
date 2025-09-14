import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.distributions import Categorical
from test import Example

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true", help="Load an existing model")
parser.add_argument("--save", action="store_true", help="Save the model after training")
parser.add_argument("--model", type=str, default="ac_ball_joint_balance/model.pt")
parser.add_argument("--lr", type=float, default=0.01)  # Learning rate
parser.add_argument("--episodes", type=int, default=800)  # Number of training episodes
parser.add_argument("--gamma", type=float, default=0.99)  # Discount factor
parser.add_argument("--run", action="store_true", help="Run the simulation")
args = parser.parse_args()

env = Example()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Actor network
class Actor(nn.Module):
    def __init__(self, in_size, out_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(in_size, 128)
        self.linear2 = nn.Linear(128, out_size)
        self.dropout = nn.Dropout(0.7)
        self.softmax = nn.Softmax(dim=1)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.softmax(self.linear2(x))
        return x


# Critic network
class Critic(nn.Module):
    def __init__(self, in_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(in_size, 128)
        self.linear2 = nn.Linear(128, 1)

        self.value_episode = []
        self.value_history = Variable(torch.Tensor()).to(device)

    def forward(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Combined module
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        value = self.critic(x)
        policy = self.actor(x)
        return value, policy


class Runner:
    def __init__(
        self,
        actor,
        critic,
        a_optimizer,
        c_optimizer,
        gamma=0.99,
        logs="ac_ball_joint_balance",
    ):
        self.actor = actor
        self.critic = critic
        self.a_opt = a_optimizer
        self.c_opt = c_optimizer
        self.gamma = gamma
        self.logs = logs
        self.writer = SummaryWriter(logs)
        self.entropy = 0

    def env_step(self, action):
        state, reward, done = env.step(action)
        return (
            torch.FloatTensor([state]).to(device),
            torch.FloatTensor([reward]).to(device),
            done,
        )

    def select_action(self, state):
        probs = self.actor(state)
        c = Categorical(probs)
        action = c.sample()

        if self.actor.policy_history.dim() != 0:
            self.actor.policy_history = torch.cat(
                [self.actor.policy_history, c.log_prob(action)]
            )
        else:
            self.actor.policy_history = c.log_prob(action)

        return action

    def estimate_value(self, state):
        pred = self.critic(state).squeeze(0)
        if self.critic.value_history.dim() != 0:
            self.critic.value_history = torch.cat([self.critic.value_history, pred])
        else:
            self.critic.policy_history = pred

    def update_a2c(self):
        R = 0
        q_vals = []

        for r in self.actor.reward_episode[::-1]:
            R = r + self.gamma * R
            q_vals.insert(0, R)

        q_vals = torch.FloatTensor(q_vals).to(device)
        values = self.critic.value_history
        log_probs = self.actor.policy_history
        advantage = q_vals - values

        self.c_opt.zero_grad()
        critic_loss = 0.0005 * advantage.pow(2).mean()
        critic_loss.backward()
        self.c_opt.step()

        self.a_opt.zero_grad()
        actor_loss = (-log_probs * advantage.detach()).mean() + 0.001 * self.entropy
        actor_loss.backward()
        self.a_opt.step()

        self.actor.reward_episode = []
        self.actor.policy_history = Variable(torch.Tensor()).to(device)
        self.critic.value_history = Variable(torch.Tensor()).to(device)

        return actor_loss, critic_loss

    def train(self, episodes, smooth=10):
        smoothed_reward = []
        max_rewards = 0

        for episode in range(episodes):
            rewards = 0
            state = env.reset()
            self.entropy = 0
            done = False

            for _ in range(1000):
                env.render()
                self.estimate_value(state)

                policy = self.actor(state).cpu().detach().numpy()
                action = self.select_action(state)

                e = -np.sum(np.mean(policy) * np.log(policy))
                self.entropy += e

                state, reward, done = env.step(action.data[0].item())
                rewards += reward
                self.actor.reward_episode.append(reward)
                if done:
                    break

            smoothed_reward.append(rewards)
            if len(smoothed_reward) > smooth:
                smoothed_reward = smoothed_reward[-1 * smooth : -1]

            a_loss, c_loss = self.update_a2c()

            self.writer.add_scalar("Critic Loss", c_loss, episode)
            self.writer.add_scalar("Actor Loss", a_loss, episode)
            self.writer.add_scalar("Reward", rewards, episode)

            if rewards >= 700 and rewards > max_rewards:
                max_rewards = rewards
                ac = ActorCritic(self.actor, self.critic)
                torch.save(ac.state_dict(), f"{self.logs}/model_{episode}.pt")

            if episode % 25 == 0:
                self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), episode)

                print(
                    f"Episode {episode} \t Final Reward {rewards:.2f} \t Average Reward: {np.mean(smoothed_reward):.2f}"
                )

    def save(self):
        ac = ActorCritic(self.actor, self.critic)
        torch.save(ac.state_dict(), f"{self.logs}/model.pt")


    def run(self):
        state = env.reset()

        for _ in range(1200):
            env.render()
            action = self.select_action(state)
            state, reward, done = env.step(action.data[0].item())


def main():
    # observation space: 8
    # action space:  (len(Example.actions))
    actor = Actor(8, len(env.actions)).to(device)
    critic = Critic(8).to(device)
    ac = ActorCritic(actor, critic)

    if args.load:
        ac.load_state_dict(torch.load(args.model))
        actor = ac.actor
        critic = ac.critic

    a_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    c_optimizer = optim.Adam(critic.parameters(), lr=args.lr)

    runner = Runner(
        actor,
        critic,
        a_optimizer,
        c_optimizer,
    )

    if args.run:
        runner.run()
    else:
        print("Training Beginning ...")
        runner.train(args.episodes)

    if args.save:
        print("Saving model...")
        runner.save()


if __name__ == "__main__":
    main()
