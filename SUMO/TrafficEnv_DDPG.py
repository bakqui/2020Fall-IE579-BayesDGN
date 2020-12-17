import traci
from sumolib import checkBinary
import numpy as np

import pandas as pd
import xml.etree.ElementTree as et

class TrafficEnv:

    def __init__(self, sumo_name, mode='binary'):

        if mode == 'gui':
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')

        self.sumo_name = sumo_name
        self.sumoFile = self.sumo_name + '.sumocfg'
        self.sumoCmd = [self.sumoBinary, "-c", self.sumoFile]

        self.time = None
        self.cycle_time = 5

    def reset(self):

        traci.start(self.sumoCmd)
        traci.simulationStep()
        self.time = 0

        return self.get_state()

    def step(self, action):

        while traci.trafficlight.getPhase("0") == 1 or traci.trafficlight.getPhase("0")  == 3 :
            traci.simulationStep()
            self.time += 1

        traci.trafficlight.setPhaseDuration("0", int(action))

        for _ in range(int(action)+1):
            traci.simulationStep()
            self.time += 1

            if self.get_done():
                break

        state = self.get_state()
        reward = self.get_reward()
        done = self.get_done()

        return state, reward, done

    def close(self):
        traci.close()
        self.time = 0

    def get_state(self):

        num_vehicle = []
        num_waiting = []
        average_velocity = []

        lane_list = traci.trafficlight.getControlledLanes("0")

        for lane_ID in lane_list:

            edge_ID = traci.lane.getEdgeID(lane_ID)

            num_vehicle.append(traci.edge.getLastStepVehicleNumber(edge_ID))
            num_waiting.append(traci.edge.getLastStepHaltingNumber(edge_ID))

        phase = [0 for _ in range(4)]
        for light_ID in traci.trafficlight.getIDList():

            phase[traci.trafficlight.getPhase(light_ID)] = 1

        state = num_vehicle + num_waiting + phase
        state = np.array(state)

        return state

    def get_reward(self):

        num_waiting = []
        for edge_ID in traci.edge.getIDList():
            num_waiting.append(traci.edge.getLastStepHaltingNumber(edge_ID))
        
        reward = - sum(num_waiting)

        return reward

    def get_done(self):

        return traci.simulation.getMinExpectedNumber() == 0