import numpy as np
from abc import ABC, abstractmethod

'''
    All mobility models defined here

    author: Zubow
'''
class AbstractMobility(ABC):
    '''
    Base class for all mobile models

    author: Zubow
    '''
    def __init__(self, node_id, pos, collect_history=True):
        self.node_id = node_id
        self.pos = np.array(pos)
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.last_update = 0

        self.collect_history = collect_history

        if self.collect_history:
            self.pos_history = {} # time -> pos
            self.velo_history = {}  # time -> velocity


    def update_pos(self, sim_time, pos, velocity, hit_wall):
        assert sim_time >= self.last_update
        self.last_update = sim_time
        self.pos = pos
        self.velocity = velocity
        self.hit_wall = hit_wall

        if self.collect_history:
            self.pos_history[sim_time] = pos
            self.velo_history[sim_time] = velocity


    @abstractmethod
    def get_pos_at(self, req_time):
        pass


    @abstractmethod
    def get_velo_at(self, req_time):
        pass


    @abstractmethod
    def get_next_direction_angle(self):
        pass


    @abstractmethod
    def check_set_new_velocity(self, sim_time, distance):
        pass


    def print(self):
        print(f'Mobility:: Id: {self.node_id}, t: {self.last_update}, pos: {self.pos}, velocity: {self.velocity}')


class ConstantMobility(AbstractMobility):
    '''
    Just static placement of nodes. The position is set from ns3.
    '''
    def __init__(self, node_id, pos, collect_history=True):
        super(ConstantMobility, self).__init__(node_id, pos, collect_history)


    def get_next_direction_angle(self):
        return 0.0


    def check_set_new_velocity(self, sim_time, distance):
        pass


    def get_pos_at(self, req_time):
        return self.pos


    def get_velo_at(self, req_time):
        return self.velocity


class RandomWalkMobility(AbstractMobility):
    # modes:
    MODE_WALL = 1 # constant speed & duration until hitting the wall
    MODE_TIME = 2
    MODE_DISTANCE = 3

    # speed:
    SPEED_UNIFORM = 1
    SPEED_CONSTANT = 2
    SPEED_NORMAL = 3

    # direction
    DIRECTION_UNIFORM = 1
    DIRECTION_CONSTANT = 2
    DIRECTION_NORMAL = 3

    '''
    A random walk mobility model with different modes and configurations.
    '''
    def __init__(self, node_id, pos, mode, mode_params, speed, speed_params, direction, direction_params, collect_history=True):
        super(RandomWalkMobility, self).__init__(node_id, pos, collect_history)
        self.mode = mode
        self.mode_params = mode_params
        self.speed = speed
        self.speed_params = speed_params
        self.direction = direction
        self.direction_params = direction_params

        if self.mode == RandomWalkMobility.MODE_WALL:
            pass
        elif self.mode == RandomWalkMobility.MODE_TIME:
            self.time_period = self.mode_params
        elif self.mode == RandomWalkMobility.MODE_DISTANCE:
            self.distance_period = self.mode_params

        # init with some speed and direction
        self._set_new_velocity()


    def get_pos_at(self, req_time):
        return self.pos_history[req_time]


    def get_velo_at(self, req_time):
        return self.velo_history[req_time]


    def _set_new_velocity(self, ts=0):
        speed = self._get_next_speed()
        direction_rad = self.get_next_direction_angle()
        # Calculate new velocity
        self.velocity = np.array([np.cos(direction_rad) * speed, np.sin(direction_rad) * speed, 0.0])
        self.last_update_velocity = ts # MODE_TIME
        self.segment_distance = 0.0 # MODE_DISTANCE


    def check_set_new_velocity(self, sim_time, distance):
        if self.mode == RandomWalkMobility.MODE_WALL:
            return
        elif self.mode == RandomWalkMobility.MODE_TIME:
            if sim_time - self.last_update_velocity >= self.time_period:
                self._set_new_velocity(sim_time)
        elif self.mode == RandomWalkMobility.MODE_DISTANCE:
            self.segment_distance += distance
            if self.segment_distance >= self.distance_period:
                self._set_new_velocity(sim_time)


    def _get_next_speed(self):
        if self.speed == RandomWalkMobility.SPEED_UNIFORM:
            return np.random.uniform(self.speed_params[0], self.speed_params[1])
        elif self.speed == RandomWalkMobility.SPEED_CONSTANT:
            return self.speed_params[0]
        elif self.speed == RandomWalkMobility.SPEED_NORMAL:
            return np.random.normal(self.speed_params[0], self.speed_params[1])


    def get_next_direction_angle(self):
        if self.direction == RandomWalkMobility.DIRECTION_UNIFORM:
            return np.random.uniform(self.direction_params[0], self.direction_params[1])
        elif self.direction == RandomWalkMobility.DIRECTION_CONSTANT:
            return self.direction_params[0]
        elif self.direction == RandomWalkMobility.DIRECTION_NORMAL:
            return np.random.normal(self.direction_params[0], self.direction_params[1])
