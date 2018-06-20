import pickle
import numpy as np
import pandas as pd

# 优化目标：点击量
obj_type = "clk"

N = 1000
c0 = 1 / 32
max_bid = 300
reduce_dim = 1

dataPath = "../data/"
projectPath = dataPath + "rlb-dp/"
data_path = dataPath + "ipinyou-data/"
ipinyou_camps = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]


def split_train_test_data(data_dir, train_size):
	df = pd.read_csv(data_dir + "/test.theta.txt", sep = ' ')
	train_num = int(len(df) * train_size)
	df_train = df.ix[:train_num, :]
	df_test = df.ix[train_num:, :]
	df_train.to_csv(data_dir + "/train_data.txt", index=False, sep=' ')
	df_test.to_csv(data_dir + "/test_data.txt", index=False, sep=' ')
	return len(df_train), len(df_test)


def calc_m_pdf(m_counter, laplace=1):

	# m(delta, x): market price p.d.f. given by x,
	# where delta is market price, x is feature vector, and m is prob
	# if a(bid) > delta, then win the auction else lose
	# m_pdf 是市场bid_price的分布。由实际市场获胜价格统计得出。

	m_pdf = [0] * len(m_counter)
	sum = 0
	for i in range(0, len(m_counter)):
		sum += m_counter[i]
	for i in range(0, len(m_counter)):
		m_pdf[i] = (m_counter[i] + laplace) / (
			sum + len(m_counter) * laplace)
	return m_pdf

def run_lb(camp_info, train_dir, test_dir, N, B):

	def get_opt_base_bid():

		bid_step = 5
		bid_performance = {}
		opt_bid = bid_step
		opt_obj = 0
		for bid in range(bid_step, max_bid + bid_step, bid_step):
			auction_in_train = open(train_dir, "r")
			(auction, imp, clk, cost) = \
				run_bid_process(auction_in_train, N, B, theta_avg, bid)

			bid_performance[bid] = [auction, imp, clk, cost]
			if clk > opt_obj:
				opt_bid = bid
				opt_obj = clk

			print('Linear Bid Algo:', bid, bid_performance[bid])

		print('Linear Bid Algo:', opt_bid, bid_performance[opt_bid], 'selected')
		return opt_bid

	def run_bid_process(auction_in, N, B, theta_avg, base_bid):
		auction = 0
		imp = 0
		clk = 0
		cost = 0
		episode = 1
		n = N
		b = B
		for line in auction_in:
			line = line.strip().split(" ")
			click = int(line[0])
			price = int(line[1])
			theta = float(line[2])
			a = 1.0 * np.log((theta + theta_avg) / theta_avg) * base_bid
			a = min(int(a), min(b, max_bid))
			if a >= price:
				# print(base_bid, a, np.log(theta / theta_avg))
				imp += 1
				if click == 1:
					clk += 1
				b -= price
				cost += price
			n -= 1
			auction += 1
			if n == 0:
				episode += 1
				n = N
				b = B
		return auction, imp, clk, cost

	theta_avg = camp_info["clk_train"] / camp_info["imp_train"]
	opt_bid = get_opt_base_bid()
	auction_in_test = open(test_dir, "r")
	(auction, imp, clk, cost) = \
		run_bid_process(auction_in_test, N, B, theta_avg, opt_bid)

	return auction, imp, clk, cost


def calc_optimal_value_function_with_approximation_i(camp_info, N, B, m_pdf):

	# Value-Iteration
	# Dynamic Programming 方法，计算 N * B 大小的2D Q-table

	# calc avg_theta, and in this exp, theta is ctr
	theta_avg = camp_info["clk_train"] / camp_info["imp_train"]
	V = [0] * ((B + 1 + reduce_dim) // reduce_dim)
	nV = [0] * ((B + 1 + reduce_dim) // reduce_dim)
	V_max = 0
	V_inc = 0

	rlb_q_table = [V]

	for b in range(0, max_bid + 1):
		V_inc += m_pdf[b] * theta_avg
	for n in range(1, N):
		# 论文 part4 - Dynamic Programming Solution
		# a(t, b, x) = argmax { sigma[delta = 0 ~ a] m_pdf(delta, x) * g(delta)} with 0<= a <= b
		# 优化目标为g(delta) 与 x轴围成的面积。
		# g(delta) = theta(x) + V(t-1, b-delta) - V(t-1, b)
		# a(t, b, x) = b if g(b) >= 0
		# 			   A if g(b) < 0 [A satisfies g(A) >= 0 and g(A) < 0]
		a = [0] * (B + 1)
		bb = B - 1
		for b in range(B, 0, -1 * reduce_dim):
			while bb >= 0 and (V[bb // reduce_dim] - V[b // reduce_dim]) + theta_avg>= 0:
				bb -= 1
			if bb < 0:
				a[b] = min(max_bid, b)
			else:
				a[b] = min(max_bid, b - bb - 1)

		V_max = V_max + V_inc
		for b in range(1, B + 1, reduce_dim):
			nV[b // reduce_dim] = V[b // reduce_dim]
			for delta in range(0, a[b] + 1):
				nV[b // reduce_dim] += m_pdf[delta] * (theta_avg + (V[(b - delta) // reduce_dim] -
													  V[b // reduce_dim]))
			if abs(nV[b // reduce_dim] - V_max) < 1e-10:
				for bb in range(b + 1, B + 1):
					nV[bb // reduce_dim] = V_max
				break
		V = nV[:]
		rlb_q_table.append(V)

	return rlb_q_table


def run_rlb(Q_table, auction_in, N, B):

	def get_bid(n, b, theta):
		a = 0
		for delta in range(1, min(b, max_bid) + 1):
			if theta + (Q_table[n - 1][(b - delta) // reduce_dim] -
						Q_table[n - 1][b // reduce_dim]) >= 0:
				a = delta
			else:
				break
		return a

	auction = 0
	imp = 0
	clk = 0
	cost = 0
	episode = 1
	n = N
	b = B

	for line in auction_in:
		line = line[:len(line) - 1].split(" ")
		click = int(line[0])
		price = int(line[1])
		theta = float(line[2])
		a = get_bid(n, b, theta)
		a = min(int(a), min(b, max_bid))

		if a >= price:
			imp += 1
			if click == 1:
				clk += 1
			b -= price
			cost += price
		n -= 1
		auction += 1

		if n == 0:
			episode += 1
			n = N
			b = B

	return auction, imp, clk, cost


def print_output(setting, auction, imp, clk, cost):
	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = clk
	log = "{:<30}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)


def main():

	log = "{:<30}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}" \
		.format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC")
	print(log)

	for camp in ipinyou_camps:

		# Basic info of that campaign
		camp_info = pickle.load(open(data_path + camp + "/info.txt", "rb"))

		# 分离train_test_data
		train_num, test_num = split_train_test_data(data_path + camp, 0.3)

		# auction log in format [is_click, wining_bid, pCtr]
		auction_in_train = open(data_path + camp + "/train_data.txt", "r")
		auction_in_test = open(data_path + camp + "/test_data.txt", "r")

		# Budget = Avg_CPM * c0 * N
		B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)
		print('train budget:', B, B * train_num / 1000)
		print('test budget:', B, B * test_num / 1000)

		# -----------------------------------------
		# Calculate m_pdf
		# -----------------------------------------
		m_pdf = calc_m_pdf(camp_info["price_counter_train"])

		# -----------------------------------------
		# Linear Bidding Algo - LB
		# -----------------------------------------
		setting = "camp={}, algo={}".format(camp, "lb")
		(auction, imp, clk, cost) = run_lb(camp_info, data_path + camp + "/train_data.txt",
			   data_path + camp + "/train_data.txt", N, B)
		print_output(setting, auction, imp, clk, cost)

		# -----------------------------------------
		# Reinforcement Learning Bidding Algo - RLB
		# -----------------------------------------
		setting = "camp={}, algo={}".format(camp, "rlb")
		rlb_q_table = calc_optimal_value_function_with_approximation_i(camp_info, N, B, m_pdf)
		(auction, imp, clk, cost) = run_rlb(rlb_q_table, auction_in_test, N, B)
		print_output(setting, auction, imp, clk, cost)


if __name__ == "__main__":
	main()



