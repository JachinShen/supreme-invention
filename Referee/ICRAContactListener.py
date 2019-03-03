from Box2D.b2 import contactListener


class ICRAContactListener(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
        self.collision_bullet_robot = []
        self.collision_bullet_wall = []
        self.collision_robot_wall = []

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        #print(u1, u2)
        if type(u1) != str or type(u2) != str:
            return
        u1_type = u1.split("_")[0]
        u2_type = u2.split("_")[0]
        if u1_type == "bullet" and u2_type == "robot":
            self.collision_bullet_robot.append((u1, u2))
        if u2_type == "bullet" and u1_type == "robot":
            self.collision_bullet_robot.append((u2, u1))
        if u1_type == "bullet" and u2_type == "wall":
            self.collision_bullet_wall.append(u1)
        if u2_type == "bullet" and u1_type == "wall":
            self.collision_bullet_wall.append(u2)
        if u1_type == "robot" and u2_type == "wall":
            self.collision_robot_wall.append(u1)
        if u2_type == "robot" and u1_type == "wall":
            self.collision_robot_wall.append(u2)

    def PostSolve(self, contact, impulse):
        pass

    def clean(self):
        self.collision_bullet_robot = []
        self.collision_bullet_wall = []
        self.collision_robot_wall = []
