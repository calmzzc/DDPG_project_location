import math
import numpy as np


class Line:
    def __init__(self):
        self.punishment_indicator = -10
        self.para_a = 0.76
        self.para_b = 0.00636
        self.para_c = 0.000115
        self.weight = 536  # 单位为t
        self.n_states = 2
        # self.n_actions = 3
        self.delta_t = 1
        self.distance = 1600
        self.scheduled_time = 100
        self.time_dim = self.scheduled_time / self.delta_t
        self.ave_dis = self.distance / self.time_dim  # 平均位移
        self.ave_vel = self.distance / self.scheduled_time  # 平均速度
        self.acc = 0.5
        self.action_space = 1
        self.delta_location = 50
        self.locate_dim = self.distance / self.delta_location
        self.ave_time = self.scheduled_time / self.locate_dim
        self.limit_speed = {0: 25, 500: 25, 1000: 25, 1500: 25, 2000: 25, 2500: 83.3, 3000: 83.3, 3500: 83.3,
                            4000: 83.3, 4500: 83.3,
                            5000: 83.3,
                            5500: 83.3, 6000: 83.3, 6500: 83.3, 7000: 83.3, 7500: 83.3, 8000: 83.3,
                            8500: 83.3, 9000: 83.3, 9500: 83.3, 10000: 83.3, 10500: 83.3, 11000: 83.3, 11500: 83.3,
                            12000: 83.3, 12500: 83.3,
                            13000: 83.3,
                            13500: 83.3, 14000: 83.3, 14500: 83.3, 15000: 70, 15500: 70, 16000: 70, 16500: 70,
                            17000: 70, 17500: 83.3,
                            18000: 83.3,
                            18500: 83.3,
                            19000: 83.3, 19500: 83.3, 20000: 83.3, 20500: 83.3, 21000: 83.3, 21500: 83.3, 22000: 83.3,
                            22500: 83.3, 23000: 83.3,
                            23500: 50,
                            24000: 50,
                            24500: 50, 25000: 50, 27500: 50}

    def calc_power(self, b_velocity, c_velocity):
        if c_velocity >= b_velocity:
            t_power = self.weight * (3.6 * 3.6 * c_velocity * c_velocity - 3.6 * 3.6 * b_velocity * b_velocity) / (
                    93312 * 1.2)
            f_power = self.weight * 9.8 * (
                    self.para_a + self.para_b * c_velocity * 3.6 + self.para_c * 3.6 * 3.6 * c_velocity * c_velocity) * (
                              c_velocity * 3.6 / 2 + b_velocity * 3.6 / 2) * abs(self.delta_location) / (
                              c_velocity / 2 + b_velocity / 2) / (12960000 * 1.2)
        else:
            t_power = 0
            f_power = self.weight * 9.8 * (
                    self.para_a + self.para_b * c_velocity * 3.6 + self.para_c * 3.6 * 3.6 * c_velocity * c_velocity) * (
                              c_velocity * 3.6 / 2 + b_velocity * 3.6 / 2) * abs(self.delta_location) / (
                              c_velocity / 2 + b_velocity / 2) / (12960000 * 1.2)
        return t_power, f_power

    def calc_power2(self, action, c_velocity):
        total_f = action * self.weight  # 单位为KN
        fric_f = (
                         self.para_a + self.para_b * c_velocity * 3.6 + self.para_c * c_velocity * 3.6 * c_velocity * 3.6) * 9.8 * self.weight / 1000  # 单位为KN
        trac_f = total_f + fric_f
        trac_e = trac_f * self.delta_location / 3600  # 单位为KWH
        trac_e = abs(trac_e)
        return trac_e

    def check_punishment(self, index, c_velocity):
        limit_speed_location = list(self.limit_speed.keys())
        for i in range(1, 51):
            if (limit_speed_location[i - 1] <= index * self.delta_location) & (
                    limit_speed_location[i] >= index * self.delta_location):
                index = i
                if c_velocity >= self.limit_speed[(index - 1) * 500]:
                    return 1
                else:
                    return 0

    def reset(self):
        # state = [0, 0]
        state = np.zeros(2)
        state[0] = np.array(0).reshape(1)
        state[1] = np.array(0).reshape(1)
        return state

    def eval_reset(self):
        state = np.zeros(2)
        state[0] = np.array(np.random.random()).reshape(1)
        state[1] = np.array(np.random.random()).reshape(1)
        return state

    def step2(self, total_power, state, action, index):
        # temp_location = state[0]
        time = np.array(state[0]).reshape(1)
        # location = state[0]
        temp_velocity = np.array(state[1]).reshape(1)
        # velocity = state[1]
        a_flag = 0

        action = self.action(action)
        if 0 <= index * self.delta_location < 0.2 * self.distance:
            action = action * self.acc
        elif 0.2 * self.distance <= index * self.delta_location < 0.65 * self.distance:
            action = ((action - 1) / 1) * self.acc
        else:
            action = -action * self.acc
        # action = ((action - 1) / 1) * self.acc
        action = np.array(action).reshape(1)
        temp_square_velocity = temp_velocity * temp_velocity + 2 * action * self.delta_location
        if temp_square_velocity <= 0:
            temp_square_velocity = np.array(0.01).reshape(1)
        velocity = np.sqrt(temp_square_velocity)
        if velocity <= 0:
            velocity = np.array(0.1).reshape(1)
        elif velocity >= 30:
            velocity = np.array(30).reshape(1)

        time = time + self.delta_location / (velocity / 2 + temp_velocity / 2)

        # tc_location = location

        state[0] = time
        state[1] = velocity

        t_power, f_power = self.calc_power(temp_velocity, velocity)
        total_power = total_power + f_power + t_power
        # t_power = self.calc_power2(action, velocity)
        total_power = total_power + t_power + f_power
        punishment_flag = self.check_punishment(index, velocity)

        beta = 0.9
        gama = 0.05
        if (index < self.locate_dim) & (time > self.scheduled_time):
            done = 2
            reward = np.array(-50).reshape(1)
        elif index == self.locate_dim:
            if time > self.scheduled_time:
                delta = 1
            else:
                delta = 0
            if punishment_flag:
                reward = np.array(-10).reshape(1)
                if reward < -300:
                    done = 2
                else:
                    done = 1
            else:
                reward = - 0.01 * t_power - 0.01 * f_power - beta * velocity - gama * abs(
                    time - self.scheduled_time) - delta * 5
                if reward < -300:
                    done = 2
                else:
                    done = 1
        else:
            done = 0
            if time > self.scheduled_time:
                delta = 1
            else:
                delta = 0
            if punishment_flag:
                reward = np.array(-10).reshape(1)
                # reward = - 0.000 * t_power - 0.000 * f_power - beta * abs(
                #     velocity - (self.distance - index * self.delta_location) / abs(
                #         self.scheduled_time - time) + 1) + self.punishment_indicator
            else:
                temp = (self.distance - index * self.delta_location) / abs((self.scheduled_time - time) + 1)
                if temp > 100:
                    temp = 100
                reward = - 0.01 * t_power - 0.01 * f_power - beta * temp - delta * 5
                # reward = - 0.01 * t_power - 0.01 * f_power - beta * abs(
                #     velocity - (self.distance - index * self.delta_location) / abs(
                #         self.scheduled_time - time) + 1) - delta * 5

        return state, reward, done, time, velocity, total_power, action

    def step(self, total_power, state, action, index):
        # temp_location = state[0]
        time = np.array(state[0]).reshape(1)
        # location = state[0]
        temp_velocity = np.array(state[1]).reshape(1)
        # velocity = state[1]
        a_flag = 0

        action = self.action(action)
        if 0 <= index * self.delta_location < 0.25 * self.distance:
            action = action * self.acc
        elif 0.25 * self.distance <= index * self.delta_location < 0.7 * self.distance:
            action = ((action - 1) / 1) * self.acc
        else:
            action = -action * self.acc
            # action = ((action - 1) / 1) * self.acc
        # action = ((action - 1) / 1) * self.acc
        action = np.array(action).reshape(1)
        temp_square_velocity = temp_velocity * temp_velocity + 2 * action * self.delta_location
        if temp_square_velocity <= 1:
            temp_square_velocity = np.array(1).reshape(1)
        velocity = np.sqrt(temp_square_velocity)
        if velocity <= 0:
            velocity = np.array(1).reshape(1)
        elif velocity >= 30:
            velocity = np.array(30).reshape(1)

        time = time + self.delta_location / (velocity / 2 + temp_velocity / 2)
        temp_time = self.delta_location / (velocity / 2 + temp_velocity / 2)  # 每一个位移间隔的时间

        # tc_location = location

        state[0] = time
        state[1] = velocity

        t_power, f_power = self.calc_power(temp_velocity, velocity)
        total_power = total_power + f_power + t_power
        # t_power = self.calc_power2(action, velocity)
        punishment_flag = self.check_punishment(index, velocity)

        beta = 0.75
        gama = 1
        if index == self.locate_dim:
            if abs(time - self.scheduled_time) <= 5:
                # delta = 10 / abs(time - self.scheduled_time)
                delta = 100
            else:
                delta = 0
            if self.scheduled_time - time < -200:
                a = -200
            elif self.scheduled_time - time > 0:
                a = -self.scheduled_time + time
            else:
                a = -1 * time + 1 * self.scheduled_time
            if punishment_flag:
                reward = -0.001 * total_power - 5 * velocity + gama * a + delta * 1
                if reward < -300:
                    done = 1
                else:
                    done = 1
            else:
                reward = -0.001 * total_power - 5 * velocity + gama * a + delta * 1
                if reward < -300:
                    done = 1
                else:
                    done = 1
        else:
            done = 0
            if time > self.scheduled_time:
                delta = 0
            else:
                delta = 0
            if punishment_flag:
                reward = -0.1 * t_power - 0.1 * f_power - 0.9 * abs(
                    temp_time - self.ave_time) + self.punishment_indicator
                # temp = (self.distance - index * self.delta_location) / abs((self.scheduled_time - time) + 1)
                # if temp > 100:
                #     temp = 100
                # reward = - 0.01 * t_power - 0.01 * f_power - beta * abs(velocity - temp) - delta * 5 - 0 * (
                #         self.locate_dim - index) + self.punishment_indicator
            else:
                reward = -0.1 * t_power - 0.1 * f_power - 0.9 * abs(temp_time - self.ave_time)
                # temp = (self.distance - index * self.delta_location) / abs((self.scheduled_time - time) + 1)
                # if temp > 100:
                #     temp = 100
                # reward = - 0.01 * t_power - 0.01 * f_power - beta * abs(velocity - temp) - delta * 5 - 0 * (
                #         self.locate_dim - index)

        return state, reward, done, time, velocity, total_power, action

    def action(self, action):
        low_bound = 0
        upper_bound = 2
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action
