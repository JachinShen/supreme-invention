from battlefield.body.robot import Robot
from utils import *
BOX_BUFF_RED = (5.8, 1.25, 1.0, 1.0)  # (x, y, w, h)
BOX_BUFF_BLUE = (1.2, 2.75, 1.0, 1.0)
# Time of defentive buff
TIME_BUFF_INIT = 30
# Time that a robot needs to stay in a buff area to activate defentive buff
TIME_BUFF_TRIGGER = 5

class BuffArea(object):
    def __init__(self):
        self.__buff_area = [
            SingleBuffArea(BOX_BUFF_RED, GROUP_RED, COLOR_LIGHT_RED),
            SingleBuffArea(BOX_BUFF_BLUE, GROUP_BLUE, COLOR_LIGHT_BLUE)
        ]
        self.__t_last = 0.0

    def detect(self, robot: Robot, t_now):
        if(robot.buff_left_time > 0):
            robot.buff_left_time -= t_now - self.__t_last
        robot.buff_left_time = max(0, robot.buff_left_time)
        for buff in self.__buff_area:
            buff.detect(robot, t_now)
        self.__t_last = t_now

    def render(self, gl):
        for buff in self.__buff_area:
            buff.render(gl)

    def get_single_buff(self, group):
        if group is GROUP_RED:
            return self.__buff_area[0]
        elif group is GROUP_BLUE:
            return self.__buff_area[1]
        else:
            print("Unknown group!!!")
            return None


class SingleBuffArea(object):
    def __init__(self, box, group, color):
        self.__box = box
        self.__t_stay = 0
        #self.maxStayTime = 0.0
        self.__group = group
        self.__color = color
        self.__t_last = 0.0
        # True: The buff area has been activated and keep invalid until next minute
        self.__activated = False

    def detect(self, robot: Robot, t_now):
        # 0, 1, 2min refresh
        if(int(t_now) % 60 == 0):
            self.__activated = False
            self.__t_stay = 0

        if self.__activated:
            return

        if robot.group != self.__group:
            return

        if self._if_in_area(robot.get_pos(), self.__box):
            self.__t_stay += t_now - self.__t_last
            if self.__t_stay >= TIME_BUFF_TRIGGER:
                self.__activated = True
                robot.buff_left_time = TIME_BUFF_INIT
                self.__t_stay = 0
        else:
            self.__t_stay = 0

        self.__t_last = t_now

    @staticmethod
    def _if_in_area(point, box):
        px, py = point
        bx, by, w, h = box
        return bx <= px <= bx + w and by <= py <= by + h

    def render(self, gl):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(*self.__color)
        x, y, w, h = self.__box
        gl.glVertex3f(x, y, 0)
        gl.glVertex3f(x + w, y, 0)
        gl.glVertex3f(x + w, y + h, 0)
        gl.glVertex3f(x, y + h, 0)
        gl.glEnd()

    def get_stay_time(self):
        return self.__t_stay

