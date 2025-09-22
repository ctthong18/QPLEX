import mate
from mate.agents import GreedyCameraAgent, GreedyTargetAgent

import numpy as np

import matplotlib.pyplot as plt

import gym


# from agent import Agent


MAX_EPISODE_STEPS = 4000


def main():
	base_env = gym.make('MultiAgentTracking-v0', config = "MATE-8v8-9.yaml")
	# base_env = mate.RenderCommunication(base_env)

	# env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent())

	env: mate.MultiAgentTracking = mate.MultiTarget(base_env, camera_agent=GreedyCameraAgent())


	# camera_agents = GreedyCameraAgent().spawn(env.num_cameras)

	target_agents = GreedyTargetAgent().spawn(env.num_targets)

	print(target_agents[0])



	target_joint_observation = env.reset()


	mate.group_reset(target_agents, target_joint_observation)
	target_info = None

	run = True

	t = []

	for i in range(MAX_EPISODE_STEPS):

		target_joint_action = mate.group_step(
			env, target_agents, target_joint_observation, target_info
		)

		results = env.step(target_joint_action)

		target_joint_observation, target_team_reward, done, target_info = results

		run = env.render()
		# arr = env.render(mode='rgb_array')

		# plt.imshow(arr)
		# plt.show()
		# arr: np.ndarray

		# a = input()


		if not run or done:
			env.close()
			break

	# print(t[0][-1, ])


if __name__ == '__main__':
	main()
