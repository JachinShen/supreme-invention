BUFFAREABOX_RED = (5.8, 1.25, 1.0, 1.0) #(x, y, w, h)
BUFFAREABOX_BLUE = (1.2, 2.75, 1.0, 1.0)

COLOR_RED = (0.9, 0.4, 0.4, 1.0)
COLOR_BLUE = (0.4, 0.4, 9.0, 1.0)

FPS = 30


class AllBuffArea(object):
    def __init__(self):
        #Time that a robot needs to stay in a buff area to activate defentive buff
        self.triggerTime = 5
        #Time of defentive buff
        self.maxBuffLeftTime = 30
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
        self.maxStayTime = 0.0
        self.group = group
        self.color  = color
        self.maxBuffLeftTime = maxBuffLeftTime
        self.triggerTime = triggerTime
        self.preTime = 0.0
        self.activated = False # True: The buff area has been activated and keep invalid until next minute

    def detect(self, objects, curTime):
        #0, 1, 2min refresh
        if(int(curTime * FPS) % (60 * FPS) == 0):
            self.activated = False
            self.stayTime = {}

        if(not self.activated):
            for car in objects:
                if(self.cover(car)):
                    if(car.hull.userData not in self.stayTime.keys()):
                        self.stayTime[car.hull.userData] = 0
                    self.stayTime[car.hull.userData] += curTime - self.preTime
                else:
                    self.stayTime[car.hull.userData] = 0
        self.preTime = curTime

        self.maxStayTime = max(self.stayTime.values()) if self.stayTime else 0
        if self.stayTime and self.maxStayTime >= self.triggerTime:
            self.stayTime = {}
            self.maxStayTime = 0.0
            self.activated = True
            for car in objects:
                if(car.group == self.group):
                    car.buffLeftTime = self.maxBuffLeftTime

    def cover(self, robot):
        # print('location:{}, {}'.format(robot.hull.position.x, robot.hull.position.y))
        # print(self.box, self.isLocated((robot.hull.position.x, robot.hull.position.y), self.box))
        return self.isLocated((robot.hull.position.x, robot.hull.position.y), self.box)

    def isLocated(self, point, box):
        px, py = point
        bx, by, w, h = box
        if(px >= bx and px <= bx + w and py >= by and py <= by + h):
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

