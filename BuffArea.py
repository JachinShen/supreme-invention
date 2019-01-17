BUFFAREABOX_RED = (0.0, 3.0, 2.0, 2.0) #(x, y, w, h)
BUFFAREABOX_BLUE = (5.0, 1.0, 1.0, 1.0)

COLOR_RED = (1.0, 0, 0, 1.0)
COLOR_BLUE = (0, 0, 1.0, 1.0)

import time


class AllBuffArea(object):
    def __init__(self):
        self.triggerTime = 5
        self.maxBuffLeftTime = 10
        self.buffAreas = [BuffArea(BUFFAREABOX_RED, 'red', COLOR_RED, self.maxBuffLeftTime, self.triggerTime),
                          BuffArea(BUFFAREABOX_BLUE, 'blue', COLOR_BLUE, self.maxBuffLeftTime, self.triggerTime)]
        self.preTime = time.time()

    def detect(self, cars):
        for car in cars:
            if(car.buffLeftTime > 0):
                car.buffLeftTime -= time.time() - self.preTime
            car.buffLeftTime = max(0, car.buffLeftTime)
            # print(car.car_id, car.buffLeftTime)
        for buff in self.buffAreas:
            buff.detect(cars)
        self.preTime = time.time()

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
        self.preTime = time.time()

    def detect(self, objects):
        for car in objects:
            if(car.buffLeftTime > 0):
                continue
            if(self.cover(car)):
                if(car.car_id not in self.stayTime.keys()):
                    self.stayTime[car.car_id] = 0
                self.stayTime[car.car_id] += time.time() - self.preTime
            else:
                self.stayTime[car.car_id] = 0
        self.preTime = time.time()

        if self.stayTime and (max(self.stayTime.values())) >= self.triggerTime:
            self.stayTime = {}
            for car in objects:
                if(car.group == self.group):
                    car.buffLeftTime = self.maxBuffLeftTime

    def cover(self, car):
        return self.isLocated((car.hull.position.x, car.hull.position.y), self.box)

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

