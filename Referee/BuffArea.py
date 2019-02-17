BUFFAREABOX_RED = (0.0, 3.0, 2.0, 2.0) #(x, y, w, h)
BUFFAREABOX_BLUE = (5.0, 1.0, 1.0, 1.0)

COLOR_RED = (1.0, 0, 0, 1.0)
COLOR_BLUE = (0, 0, 1.0, 1.0)


class AllBuffArea(object):
    def __init__(self):
        #Time that a robot needs to stay in a buff area to activate defentive buff
        self.triggerTime = 5
        #Time of defentive buff
        self.maxBuffLeftTime = 10
        self.buffAreas = [BuffArea(BUFFAREABOX_RED, 'red', COLOR_RED, self.maxBuffLeftTime, self.triggerTime),
                          BuffArea(BUFFAREABOX_BLUE, 'blue', COLOR_BLUE, self.maxBuffLeftTime, self.triggerTime)]
        self.preTime = 0.0

    def detect(self, robots, curTime):
        for robot in robots:
            if(robot.buffLeftTime > 0):
                robot.buffLeftTime -= curTime - self.preTime
            robot.buffLeftTime = max(0, robot.buffLeftTime)
            # print(car.car_id, car.buffLeftTime)
        for buff in self.buffAreas:
            buff.detect(robots, curTime)
        self.preTime = curTime

    def render(self, gl):
        for buff in self.buffAreas:
            buff.render(gl)

class BuffArea(object):
    def __init__(self, box, group, color, maxBuffLeftTime, triggerTime):
        self.box = box
        self.stayTime = {}
        self.group = group
        self.color  = color
        self.maxBuffLeftTime = maxBuffLeftTime
        self.triggerTime = triggerTime
        self.preTime = 0.0

    def detect(self, objects, curTime):
        for car in objects:
            if(car.buffLeftTime > 0):
                continue
            if(self.cover(car)):
                if(car.robot_id not in self.stayTime.keys()):
                    self.stayTime[car.robot_id] = 0
                self.stayTime[car.robot_id] += curTime - self.preTime
            else:
                self.stayTime[car.robot_id] = 0
        self.preTime = curTime

        if self.stayTime and (max(self.stayTime.values())) >= self.triggerTime:
            self.stayTime = {}
            for car in objects:
                if(car.group == self.group):
                    car.buffLeftTime = self.maxBuffLeftTime

    def cover(self, robot):
        return self.isLocated((robot.hull.position.x, robot.hull.position.y), self.box)

    def isLocated(self, point, box):
        px, py = point
        bx, by, w, h = box
        if(px >= bx and px <= bx + w and py >= bx and py <= by + h):
            return True
        else:
            return False



    def render(self, gl):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(self.color[0], self.color[1], self.color[2], self.color[3])
        x, y, w, h = self.box
        gl.glVertex3f(x, y, 0)
        gl.glVertex3f(x + w, y, 0)
        gl.glVertex3f(x + w, y + h, 0)
        gl.glVertex3f(x, y + h, 0)
        gl.glEnd()

