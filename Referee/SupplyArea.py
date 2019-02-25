SUPPLYAREABOX_RED = (3.5, 4.0, 1.0, 1.0) #(x, y, w, h)
SUPPLYAREABOX_BLUE = (3.5, 0, 1.0, 1.0)

COLOR_RED = (0.8, 0, 0, 0.8)
COLOR_BLUE = (0, 0, 0.8, 0.8)

class SupplyAreas(object):
    def __init__(self):
        self.supply_area_red = SUPPLYAREABOX_RED
        self.supply_area_blue = SUPPLYAREABOX_BLUE

    def render(self, gl):
        self._render(gl, self.supply_area_red, COLOR_RED)
        self._render(gl, self.supply_area_blue, COLOR_BLUE)

    def _render(self, gl, box, color):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(color[0], color[1], color[2], color[3])
        x, y, w, h = box
        gl.glVertex3f(x, y, 0)
        gl.glVertex3f(x + w, y, 0)
        gl.glVertex3f(x + w, y + h, 0)
        gl.glVertex3f(x, y + h, 0)
        gl.glEnd()

    # def isInSupplyArea(self, object):
    #     if(object.group not in {'red', 'blue'}):
    #         return False
    #     if(object.group == 'red'):
    #         supply_area = self.supply_area_red
    #     elif(object.group == 'blue'):
    #         supply_area = self.supply_area_blue
    #
    #     x_robot, y_robot = object.hull.position.x, object.hull.position.y
    #     bx, by, w, h = supply_area
    #     if (x_robot >= bx and x_robot <= bx + w and y_robot >= by and y_robot <= by + h):
    #         return True
    #     else:
    #         return False
