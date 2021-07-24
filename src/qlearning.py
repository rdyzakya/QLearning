import random
import time
import numpy as np
from IPython.display import clear_output

class BananaHole:
	def __init__(self):
		self.player_pos = 3
		self.available_actions = 2
		self.states = 10
		self.score = 0

	def action_sample(self):
		return random.randint(0,1)

	def move(self,action):
		if action == 0:
			if self.player_pos > 0:
				self.player_pos -= 1

			if self.player_pos == 0:
				self.score -= 1
				self.player_pos = 3

		if action == 1:
			if self.player_pos < 9:
				self.player_pos += 1

			if self.player_pos == 9:
				self.score += 1
				self.player_pos = 3

		state = self.player_pos
		reward = self.reward(action)
		done = self.is_done()
		return state,reward,done,self.score

	def reward(self,move):
		return (self.player_pos + np.exp(self.score))*move

	def is_done(self):
		return self.score == 5 or self.score == -5

	def reset(self):
		self.__init__()
		return self.player_pos

	def render(self):
		print('<',end='')
		for i in range(10):
			if i == self.player_pos:
				print('P',end='')
			else:
				if i == 0:
					print('O',end='')
				elif i == 9:
					print('B',end='')
				else:
					print('-',end='')
		print('>')

def q_learning(env,n_eps,st_per_eps,learning_rate,disc_rate,explore_decay_rate,render=False):
	rewards_all_episodes = []
	exp_rate = 1
	max_exp_rate = 1
	min_exp_rate = 0.01
	q_table = np.zeros((env.states,env.available_actions))
	win = 0
	lose = 0

	for eps in range(n_eps):
		state = env.reset()
		done = False
		reward_this_eps = 0
		if render:
			print("EPISODE " , eps + 1)
			time.sleep(1)

		for step in range(st_per_eps):
			if render:
				clear_output(wait=True)
				env.render()
				print('win : ', win, ' | lose : ', lose)
				time.sleep(0.1)
			exp_rate_threshold = random.random()
			action = None
			if exp_rate_threshold > exp_rate:
				action = np.argmax(q_table[state,:])
			else:
				action = env.action_sample()
			new_state, reward, done, score = env.move(action)

			q_table[state,action] = (q_table[state,action] * (1-learning_rate) + 
				learning_rate * (reward + disc_rate * np.max(q_table[state,:])))

			state = new_state
			reward_this_eps += reward

			if done:
				if render:
					clear_output(wait=True)
					env.render()
					if score == 5:
						print("you win!")
						win += 1
					else:
						print("you lose!")
						lose += 1
					print('win : ', win, ' | lose : ', lose)
					time.sleep(0.7)
					clear_output(wait=True)
				else:
					if score == 5:
						win += 1
					else:
						lose += 1
				break

		exp_rate = min_exp_rate + (max_exp_rate - min_exp_rate) * np.exp(-explore_decay_rate * eps)
		rewards_all_episodes.append(reward_this_eps)

	return q_table,win,lose


if __name__ == '__main__':
	b = BananaHole()
	while(not b.is_done()):
		b.render()
		i = int(input())
		b.move(i)
		print(b.score)