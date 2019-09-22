
class RobotState():
    def __init__(self):
        self.health = 2000
        self.pos = [-1, -1]
        self.angle = -1
        self.velocity = [0, 0]
        self.angular = 0
        self.detect = False
        self.scan = []

class Action():
    def __init__(self):
        self.v_t = 0.0
        self.v_n = 0.0
        self.omega = 0.0
        self.shoot = 0.0
        self.supply = 0.0
