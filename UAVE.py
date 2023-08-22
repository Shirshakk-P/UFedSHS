import stable_baselines3.common.env_checker		# check if the created environment is inline with standard formats
import numpy as np
import requests
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from gym import spaces
from IPython import display

num_uavs = 5
num_users = 12

theta_u_o = 100       # Upper Limit of normal value for Oxygen Sensor 
theta_u_b = 169       # Upper Limit of normal value for Blood Pressure Sensor 
theta_u_t = 37.9      # Upper Limit of normal value for Temperature Sensor 
theta_u_r = 29        # Upper Limit of normal value for Respiratory Sensor 
theta_u_h =  139      # Upper Limit of normal value for Heart Rate Sensor 

theta_l_o = 95        # Lower Limit of normal value for Oxygen Sensor 
theta_l_b = 91        # Lower Limit of normal value for Blood Pressure Sensor
theta_l_r = 11        # Lower Limit of normal value for Respiratory Sensor 
theta_l_t = 34.1      # Lower Limit of normal value for Temperature Sensor 
theta_l_h = 51        # Lower Limit of normal value for Heart Rate Sensor 

alpha_b = 2704        # Medical Criticality value for Blood Pressure Sensor
alpha_o = 2003        # Medical Criticality value for Oxygen Sensor
alpha_t = 1071        # Medical Criticality value for Temperature Sensor
alpha_r = 1776        # Medical Criticality value for Respiratory Sensor
alpha_h = 8205        # Medical Criticality value for Heart Rate Sensor

P_avg = 1e4             # Average Transmission power of UAVs
P_d = 1.5e4             # Data Acquisition power of UAVs 

v = 15                  # Average velocity of the UAVs
kappa = 1e-28           # Computation Capacity of the UAVs
f = 1.5e6               # UAV Computation Frequency
d = 2e5                 # Data size per user
L = 5900                # Computation cycles required per byte 
r = L*d 

beta_1 = 11.5           # Price per unit Criticality of patient
beta_2 = 0.000003565    # Price per unit Joule of energy 
beta_3 = 0.0000002005   # Price per unit MB of data 
q = 1                   # UAV movement radius
B = 20e6                # 5G Transmission Bandwidth

Z_theta = 0.5           # Max threshold for non-critical condition 
psi_theta = 5           # Minimum permitted distance between two UAVs
U_theta = 4             # Maximum permitted users per UAV
E_theta = 2000          # Minimum residual energy of the UAVs
(xB,yB) = (20,20)       # Location of the Base Station
psi = 1e2               # Propulsion power of UAVs

eta_1 = 0.35            # Profit weight in RL Reward function
eta_2 = 0.25            # Energy weight in RL Reward function
e_0 = 1e5               # Initial and post-charging energy of UAVs

tau = np.random.uniform(low = 2e-2, high = 7e-2, size = (num_uavs, num_users))  # Time matrix denoting transmission time between different Users and UAVs
e_A = tau * P_d                                                                 # Constant energy per user to transfer data to UAVs

matrix = np.zeros((num_uavs, num_users))  # User-UAV Association Matrix
for col in range(num_users):
  row = np.random.choice(num_uavs)
  matrix[row, col] = 1


h_i_o = []                                # Criticality of users via the Oxygen Sensor
h_i_b = []                                # Criticality of users via the Blood Pressure Sensor
h_i_r = []                                # Criticality of users via the Respiratory Sensor
h_i_t = []                                # Criticality of users via the Temperature Sensor
h_i_h = []                                # Criticality of users via the Heart Beat Sensor

theta_b = np.random.randint(low = 91, high = 170, size = (num_users))           # Criticality value of users via the Blood Pressure Sensor
theta_t = np.random.uniform(low = 34.1, high = 37.6, size = (num_users))        # Criticality value of users via the Temperature Sensor
theta_h = np.random.randint(low = 51, high = 140, size = (num_users))           # Criticality value of users via the Heart Rate Sensor
theta_r = np.random.randint(low = 11, high = 30, size = (num_users))            # Criticality value of users via the Respiratory Sensor
theta_o = np.random.uniform(low = 95, high = 100, size = (num_users))           # Criticality value of users via the Oxygen Sensor

e_C = kappa * f * f* r                                                          # Constant energy per user to transfer data to UAVs

tau_U = np.random.uniform(low = 1.5e-2, high = 3e-2, size = (num_uavs))         # Time matrix denoting transmission time between different Users and UAVs

total_E = []                                                                    # Total energy consumed per UAV

SINR = np.random.randint(low=10, high=14, size=num_uavs)                        # SINR Value between UAV and Hospital transmissions
data = np.ones(num_users) * d                                                   # Data Array denoting amount of data to be sent 

def trans_Rate(SINR):                              # (3)
  res = []
  for i in range(num_uavs):
    val = B * math.log2(1 + SINR[i])
    res.append(val)
  return res  

M = trans_Rate(SINR)                                                            # Transmission Data Rate array for UAVs
                                                              # UAV Movement array initialization
max_energy = e_0
uav_energy_consumptions = np.zeros(num_uavs)                                    # Initial UAV energy consumptions

class UAVEnvironment(gym.Env):
    def __init__(self, num_uavs, num_users, max_energy):
      self.theta_u_o = 100       # Upper Limit of normal value for Oxygen Sensor 
      self.theta_u_b = 169       # Upper Limit of normal value for Blood Pressure Sensor 
      self.theta_u_t = 37.9      # Upper Limit of normal value for Temperature Sensor 
      self.theta_u_r = 29        # Upper Limit of normal value for Respiratory Sensor 
      self.theta_u_h =  139      # Upper Limit of normal value for Heart Rate Sensor 

      self.theta_l_o = 95        # Lower Limit of normal value for Oxygen Sensor 
      self.theta_l_b = 91        # Lower Limit of normal value for Blood Pressure Sensor
      self.theta_l_r = 11        # Lower Limit of normal value for Respiratory Sensor 
      self.theta_l_t = 34.1      # Lower Limit of normal value for Temperature Sensor 
      self.theta_l_h = 51        # Lower Limit of normal value for Heart Rate Sensor 
      
      self.num_users = num_users           # Number of Users
      self.num_uavs = num_uavs            # Number of UAVs

      self.alpha_b = 2704        # Medical Criticality value for Blood Pressure Sensor
      self.alpha_o = 2003        # Medical Criticality value for Oxygen Sensor
      self.alpha_t = 1071        # Medical Criticality value for Temperature Sensor
      self.alpha_r = 1776        # Medical Criticality value for Respiratory Sensor
      self.alpha_h = 8205        # Medical Criticality value for Heart Rate Sensor
      
      self.theta_b = np.random.randint(low = 91, high = 170, size = (self.num_users))           # Criticality value of users via the Blood Pressure Sensor
      self.theta_t = np.random.uniform(low = 34.1, high = 37.6, size = (self.num_users))        # Criticality value of users via the Temperature Sensor
      self.theta_h = np.random.randint(low = 51, high = 140, size = (self.num_users))           # Criticality value of users via the Heart Rate Sensor
      self.theta_r = np.random.randint(low = 11, high = 30, size = (self.num_users))            # Criticality value of users via the Respiratory Sensor
      self.theta_o = np.random.uniform(low = 95, high = 100, size = (self.num_users))           # Criticality value of users via the Oxygen Sensor

      self.P_avg = 1e4             # Average Transmission power of UAVs
      self.P_d = 1.5e4             # Data Acquisition power of UAVs 

      self.v = 15                  # Average velocity of the UAVs
      self.kappa = 1e-28           # Computation Capacity of the UAVs
      self.f = 1.5e6               # UAV Computation Frequency
      self.d = 2e5                 # Data size per user
      self.L = 5900                # Computation cycles required per byte 
      self.r = self.L*self.d 

      self.beta_1 = 86.5           # Price per unit Criticality of patient
      self.beta_2 = 0.000000002565    # Price per unit Joule of energy 
      self.beta_3 = 0.000000000005   # Price per unit MB of data 
      self.q = 1                   # UAV movement radius
      self.B = 20e6                # 5G Transmission Bandwidth
      self.Z_theta = 0.5           # Max threshold for non-critical condition 
      self.psi_theta = 5           # Minimum permitted distance between two UAVs
      self.U_theta = 4             # Maximum permitted users per UAV
      self.E_theta = 2000          # Minimum residual energy of the UAVs
      self.psi = 1e2               # Propulsion power of UAVs
      self.eta_1 = 0.35            # Profit weight in RL Reward function
      self.eta_2 = 0.25            # Energy weight in RL Reward function
      self.e_0 = 1e5               # Initial and post-charging energy of UAVs
      self.xB = 20                 # Base station x-coordinate
      self.yB = 20                 # Base station y-coordinate
      self.x0 = 0                  # User x-coordinate min value
      self.y0 = 0                  # User y-coordinate min value
      self.xU = 0                  # UAV x-coordinate min value
      self.yU = 0                  # UAV y-coordinate min value
      self.X0 = 100
      self.Y0 = 100
      self.XU = 100
      self.YU = 100
      self.num_uavs = num_uavs
      self.num_users = num_users
      self.max_energy = max_energy
      self.user_locations = np.round(np.random.uniform(low=0, high=100, size=(self.num_users, 2)), decimals=2)
      self.uav_locations = np.round(np.random.uniform(low=0, high=100, size=(self.num_uavs, 2)), decimals=2)
      self.uav_energy_levels = np.round(np.random.uniform(low=self.E_theta+1200, high=self.e_0, size=(self.num_uavs)), decimals=2)
      self.angular_movement = True
      self.rewards = 1.5e5

      self.action_space = spaces.MultiDiscrete([8]*self.num_uavs)
      self.uav_shape = [self.XU, self.YU, self.e_0]*self.num_uavs
      self.user_shape = [self.X0, self.Y0]*self.num_users
      self.observation_space = spaces.Box(low = np.zeros(len(self.uav_shape + self.user_shape)), high=np.array(self.uav_shape + self.user_shape), dtype=np.float64)
      self.tau = np.random.uniform(low = 2e-2, high = 7e-2, size = (self.num_uavs, self.num_users))  # Time matrix denoting transmission time between different Users and UAVs
      self.e_A = self.tau * self.P_d                                                                 # Constant energy per user to transfer data to UAVs

      self.matrix = np.zeros((self.num_uavs, self.num_users))  # User-UAV Association Matrix
      for col in range(self.num_users):
        row = np.random.choice(self.num_uavs)
        self.matrix[row, col] = 1

      self.penalty = 0.5

      self.h_i_o = []                                # Criticality of users via the Oxygen Sensor
      self.h_i_b = []                                # Criticality of users via the Blood Pressure Sensor
      self.h_i_r = []                                # Criticality of users via the Respiratory Sensor
      self.h_i_t = []                                # Criticality of users via the Temperature Sensor
      self.h_i_h = []                                # Criticality of users via the Heart Beat Sensor

      self.theta_b = np.random.randint(low = 91, high = 170, size = (num_users))           # Criticality value of users via the Blood Pressure Sensor
      self.theta_t = np.random.uniform(low = 34.1, high = 37.6, size = (num_users))        # Criticality value of users via the Temperature Sensor
      self.theta_h = np.random.randint(low = 51, high = 140, size = (num_users))           # Criticality value of users via the Heart Rate Sensor
      self.theta_r = np.random.randint(low = 11, high = 30, size = (num_users))            # Criticality value of users via the Respiratory Sensor
      self.theta_o = np.random.uniform(low = 95, high = 100, size = (num_users))           # Criticality value of users via the Oxygen Sensor
      self.e_C = self.kappa * self.f * self.f* self.r                                                          # Constant energy per user to transfer data to UAVs
      self.tau_U = np.random.uniform(low = 1.5e-2, high = 3e-2, size = (num_uavs))         # Time matrix denoting transmission time between different Users and UAVs

      self.total_E = []                                                                    # Total energy consumed per UAV

      self.SINR = np.random.randint(low=10, high=14, size=self.num_uavs)                        # SINR Value between UAV and Hospital transmissions

      self.data = np.ones(num_users) * self.d                                              # Data Array denoting amount of data to be sent 
      def trans_Rate(SINR):
      	res = []
      	for i in range(num_uavs):
        	val = self.B * math.log2(1 + SINR[i])
        	res.append(val)
      	return res 
      	
      self.M = trans_Rate(self.SINR)                                                            # Transmission Data Rate array for UAVs
      self.energy_ini = self.e_0 * np.ones(self.num_uavs)                                            # UAV Initial Energy array
      self.f = 1.5e6                                                            # UAV Movement array initialization
      self.max_energy = self.e_0
      self.uav_energy_levels = np.round(np.random.uniform(low=self.E_theta+1200, high=self.e_0, size=(self.num_uavs)), decimals=2)     # Randomly assigning UAV energy per iteration
          

    def reset(self):
       uav_shape = [self.XU, self.YU, self.e_0] * self.num_uavs
       user_shape = [self.X0, self.Y0] * self.num_users
       state = np.array(uav_shape + user_shape)
       print(self.observation_space.sample().dtype)
       print(state.dtype)
       return state

# ____________________________________________________________

    def new_position(self, current_pos, angle):                       
      new_x = current_pos[0] + q * np.cos(angle)
      new_y = current_pos[1] + q * np.sin(angle)
      return np.array([new_x, new_y])

    def get_energy(self, current_pos, new_pos):
      euclidean_dist = np.linalg.norm(new_pos - current_pos)
      energy = (psi/ v) * euclidean_dist
      return energy

    def remaining_energy(self, total_Energy, energy_ini):             # ______________________________ Remaining energy function
        res = []
        for i in range(num_uavs):
          val = energy_ini[i] - total_Energy[i]
          res.append(val)
        return res


# ______________________________________________________________________________________

        # Re-initialize the action space based on the angular movement
    def step(self, action:int):
      reward = self.rewardF(self.uav_locations, action)
      self._get_next_state(action)

      uav_movements = np.zeros((self.num_uavs, 2))
      self.uav_energy_consumptions = np.zeros(self.num_uavs) # + total_E
      self.uav_energy_levels = np.round((self.uav_energy_levels - self.uav_energy_consumptions), decimals=2)
      print("energy" ,self.uav_energy_levels)
        

      E_A = self.data_acquisition_energy(e_A, matrix)
      E_T = self.transmission_energy()
      E_C = self.computation_energy(matrix)
      E_TFL = self.FL_transmission_energy(tau_U)
      dist = self.base_dist(self.user_locations)
      E_X = self.traversal_energy(matrix, dist)
      total_E = self.total_Energy(E_A, E_T, E_C, E_TFL, E_X)
      self.uav_energy_consumptions = total_E
      self.uav_energy_levels = self.uav_energy_levels - self.uav_energy_consumptions
      new_dist = self.new_traversal_dist(self.uav_locations, self.user_locations)
      E_h = self.H_traversal_energy(matrix, new_dist)
      hB, hO, hH, hR, hT = self.health_index(theta_b, theta_o, theta_h, theta_r, theta_t)
      criticalitY = self.criticality(hB, hO, hH, hR, hT)
      Ww = self.W(criticalitY, d)
      cosT = self.cost(total_E, E_h)
      revenuE = self.revenue(Ww, matrix)
      profiT = self.profit(revenuE, cosT)
      rewards = self.rewardF(cosT, profiT)
      c_users = self.connected_users(self.matrix)

      print("UAV_energy_levels: ", self.uav_energy_levels)
      print("UAV_energy_consumptions: ", self.uav_energy_consumptions)
      print("UAV locations:", self.uav_locations)
      print("User locations:", self.user_locations)
      print("Total Energy=",total_E)
      #print("Cost=", cosT)
      #print("Revenue=", revenuE)
      print("Profit = ",profiT)
      print("Reward =",rewards)
      #print("E_H", E_h)
      #print("New dist" ,new_dist)
      print("Connected Users:", c_users)
      
      uav_sh = np.c_[self.uav_locations, self.uav_energy_levels]
      user_sh = self.user_locations
      next_state = np.concatenate((uav_sh.flatten(),user_sh.flatten()), dtype=np.float64)
      reward = sum(rewards)
      print("UAV Next State: ",next_state)
             
      # Determine if the episode is done
      done = bool(((self.uav_energy_levels - self.uav_energy_consumptions) <= E_theta).any() or (max(self.connected_users(matrix)) >= U_theta).any())   #changed
      return next_state, reward, done, {}

    def _get_next_state(self, action):
      print(action)
      locations = []
      print("Pre-actions", self.uav_locations, self.uav_energy_levels)
      for i in range(self.num_uavs):
        if action[i] == 0:
          self.uav_locations[i] = [self.uav_locations[i][0] + (self.q * math.cos(math.radians(0))), self.uav_locations[i][1] + (self.q*math.sin(math.radians(0)))]
          self.uav_energy_levels0 = self.uav_energy_levels - self.uav_energy_levels*self.q*(self.psi/self.v)
          if(self.uav_energy_levels0.all() <= self.uav_energy_levels.all()):
            self.rewards += 1000
          else:
            self.rewards -= 350
          print("Action Taken: 0")
        
        elif action[i] == 1:
          self.uav_locations[i] = [self.uav_locations[i][0] + self.q*math.cos(math.radians(45)), self.uav_locations[i][1] + self.q*math.sin(math.radians(45))]
          self.uav_energy_levels1 = self.uav_energy_levels - self.uav_energy_levels*1.414*self.q*(self.psi/self.v)
          if(self.uav_energy_levels1.all() <= self.uav_energy_levels.all()):
            self.rewards += 1000
          else:
            self.rewards -= 350
          print("Action Taken: 1")

        elif action[i] == 2:
          self.uav_locations[i] = [self.uav_locations[i][0] + self.q*math.cos(math.radians(90)), self.uav_locations[i][1] + self.q*math.sin(math.radians(90))]
          self.uav_energy_levels2 = self.uav_energy_levels - self.uav_energy_levels*self.q*(self.psi/self.v)
          if(self.uav_energy_levels2.all() <= self.uav_energy_levels.all()):
            self.rewards += 1000
          else:
            self.rewards -= 350
          print("Action Taken: 2")

        elif action[i] == 3:
          self.uav_locations[i] = [self.uav_locations[i][0] + self.q*math.cos(math.radians(135)), self.uav_locations[i][1] + self.q*math.sin(math.radians(135))]
          self.uav_energy_levels3 = self.uav_energy_levels - self.uav_energy_levels*1.414*self.q*(self.psi/self.v)
          if(self.uav_energy_levels3.all() <= self.uav_energy_levels.all()):
            self.rewards += 1000
          else:
            self.rewards -= 350
          print("Action Taken: 3")

        elif action[i] == 4:
          self.uav_locations[i] = [self.uav_locations[i][0] + self.q*math.cos(math.radians(180)), self.uav_locations[i][1] + self.q*math.sin(math.radians(180))]
          self.uav_energy_levels4 = self.uav_energy_levels - self.uav_energy_levels*self.q*(self.psi/self.v)
          if(self.uav_energy_levels4.all() <= self.uav_energy_levels.all()):
            self.rewards += 1000
          else:
            self.rewards -= 350
          print("Action Taken: 4")

        elif action[i] == 5:
          self.uav_locations[i] = [self.uav_locations[i][0] + self.q*math.cos(math.radians(225)), self.uav_locations[i][1] + self.q*math.sin(math.radians(225))]
          self.uav_energy_levels5 = self.uav_energy_levels - self.uav_energy_levels*1.414*self.q*(self.psi/self.v)
          if(self.uav_energy_levels5.all()<= self.uav_energy_levels.all()):
            self.rewards += 1000
          else:
            self.rewards -= 350
          print("Action Taken: 5")

        elif action[i] == 6:
          self.uav_locations[i] = [self.uav_locations[i][0] + self.q*math.cos(math.radians(270)), self.uav_locations[i][1] + self.q*math.sin(math.radians(270))]
          self.uav_energy_levels6 = self.uav_energy_levels - self.uav_energy_levels*self.q*(self.psi/self.v)
          if(self.uav_energy_levels6.all() <= self.uav_energy_levels.all()):
            self.rewards += 1000
          else:
            self.rewards -= 350
          print("Action Taken: 6")
          
        elif action[i] == 7:
          self.uav_locations[i] = [self.uav_locations[i][0] + self.q*math.cos(math.radians(315)), self.uav_locations[i][1] + self.q*math.sin(math.radians(315))]
          self.uav_energy_levels7 = self.uav_energy_levels - self.uav_energy_levels*1.414*self.q*(self.psi/self.v)
          if(self.uav_energy_levels7.all() <= self.uav_energy_levels.all()):
            self.rewards += 1000
          else:
            self.rewards -= 350
          print("Action Taken: 7")
          
        else:
          raise ValueError("Action value not supported:", action)

# ______________________________________________________________________Other Functions_________________________

    def health_index(self, theta_b, theta_o, theta_h, theta_r, theta_t):                        # (1)
      for i in range(num_users):         
        res = np.round(abs(((theta_u_b - theta_b[i])**2 - (theta_b[i] - theta_l_b)**2) / (theta_u_b + theta_l_b)**2), decimals=2)
        h_i_b.append(res)
      for i in range(num_users):
        res = np.round(abs(((theta_u_o - theta_o[i])**2 - (theta_o[i] - theta_l_o)**2) / (theta_u_o + theta_l_o)**2), decimals=2)
        h_i_o.append(res)
      for i in range(num_users):
        res = np.round(abs(((theta_u_h - theta_h[i])**2 - (theta_h[i] - theta_l_h)**2) / (theta_u_h + theta_l_h)**2), decimals=2)
        h_i_h.append(res)
      for i in range(num_users):
        res = np.round(abs(((theta_u_r - theta_r[i])**2 - (theta_r[i] - theta_l_r)**2) / (theta_u_r + theta_l_r)**2), decimals=2)
        h_i_r.append(res)
      for i in range(num_users):
        res = np.round(abs(((theta_u_t - theta_t[i])**2 - (theta_t[i] - theta_l_t)**2) / (theta_u_t + theta_l_t)**2), decimals=2)
        h_i_t.append(res)
      return h_i_b, h_i_o, h_i_h, h_i_r, h_i_t

    def criticality(self, hb, ho, hh, hr, ht):                        # (2)
      hb = hb * alpha_b
      ho = ho * alpha_o
      hh = hh * alpha_h
      hr = hr * alpha_r
      ht = ht * alpha_t

      res = []
      for i in range(num_users):
        val = hb[i] + ho[i] + hh[i] + hr[i] + ht[i]
        res.append(val)
      return res

    # _____________ Add a distance between uavs function

    def connected_users(self, matrix):
      res = []
      for i in range(len(matrix)):
        res.append(sum(matrix[i]))
      return res

    def Data(self, matrix):                                             # (4)
      res = []
      for i in range(num_uavs):
        val = self.d * sum(matrix[i])
        res.append(val)
      return res
    
    def transmission_energy(self):                                        # (6)
      res = []
      for i in range(num_uavs):
        val = (data[i]/M[i])*P_avg                      
        res.append(val)
      return res

    def data_acquisition_energy(self, energy, matrix):                   # (8)
      arr = np.multiply(energy, matrix)
      res = []
      return np.sum(arr, axis=1)
      
    def base_dist(self, user):                                            # (9)
      res = []
      for i in range(self.num_users):
        val = math.sqrt((user[i][0] - xB)**2 + (user[i][1] - yB)**2)
        res.append(val)
      return res

    def traversal_energy(self,matrix, dist):                             # (10)
      res = []
      for i in range(num_uavs):
        val = (sum(matrix[i]) * dist[i]) * (self.psi /self.v)
        res.append(val)
      return res
  
    def new_traversal_dist(self, uav_locations, user_locations):                     # (11)
      res = []
      for i in range(self.num_uavs):
        for j in range(self.num_users):
          if matrix[i][j] == 1:
            print(user_locations, uav_locations)
            val = math.sqrt((user_locations[j][0] - uav_locations[i][0])**2 + (user_locations[j][1]- uav_locations[i][1])**2)
            res.append(val)
      print(res)
      return res

    def H_traversal_energy(self, matrix, new_dist):                         # (12)
      res = []
      for i in range(num_uavs):
        val = (sum(matrix[i]) * new_dist[i]) * (self.psi /self.v)
        res.append(np.round(val, decimals=2))
      return res

    def computation_energy(self, matrix):                                   # (14)
      res = []
      for i in range(self.num_uavs):
        val = np.round(self.e_C * sum(matrix[i]), decimals=2)
        res.append(val)
      return res

    def FL_transmission_energy(self, tau_U):                                # (17)
      res = []
      for i in range(num_uavs):
        val = tau_U[i] * self.P_avg
        res.append(val)
      return res

    def total_Energy(self, E_A, E_T, E_C, E_TFL, E_X):                       # (18)
      res = []
      for i in range(self.num_uavs):
        val = E_A[i] + E_T[i] + E_X[i] + E_C[i] + E_TFL[i]
        res.append(np.round(val, decimals=2))
      return res

    def cost(self, total_E, E_H):                                            # (19)
      res = []
      for i in range(self.num_uavs):
        val = np.round(self.beta_1*(total_E[i] + E_H[i]), decimals=2)
        res.append(val)      
      return res 

    def W(self, criticality, d):                                          # (20)
      for i in range(len(criticality)):
        criticality[i] = self.beta_2 * criticality[i]                              
      res = np.round(np.sum([criticality, d*np.ones(num_users)], axis=0), decimals=3)
      return res 


    def revenue(self, W, matrix):                                            # (21)
      res = []
      for i in range(num_uavs):
        res.append(sum(np.multiply(W, matrix[i])))
      return res

    def profit(self, revenue, cost):                                        # (22)
      res = []
      for i in range(self.num_uavs):
        val = np.round(revenue[i] - cost[i], decimals=2)
        res.append(val)
      return res

    def rewardF(self, cost, profit):                      # _________________________ Reward Function
      for i in range(self.num_uavs):
        prof = np.multiply(self.eta_1, profit)
      for i in range(num_uavs):
        cos = np.multiply(self.eta_2,cost)
      res = []
      for i in range(self.num_uavs):
        res.append(prof[i] - cos[i])
      return res


#env1 = UAVEnvironment(5, 12, 1e5)
#stable_baselines3.common.env_checker.check_env(env1)

#num_episodes = 1
'''
def test_env(env: gym.Env) -> None:
    env.reset()
    done = False
    while not done:
      for k in range(num_episodes):
        print("\n____________EPISODE____________________________")
        action = env.action_space.sample()
        print("Action by UAVs:")
        _, _, done, _ = env.step(action)
        
        
test_env(env1)
'''
