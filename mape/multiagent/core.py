import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = 3
        self.accel = 2
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 0.2

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        self.size = 0.05

# properties of agent entities
class Agent(Entity):
    def __init__(self, iden=None):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # prev_dist
        self.prev_dist = None
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        if iden is not None:
            self.iden = iden

# multi-agent world
class World(object):
    def __init__(self, env_id):
        # list of agents and entities (can change at execution-time!)
        self.env_id = env_id
        if self.env_id == 'simple_waypoints':
            # self.intervals = [25,40,100,125]
            # self.waypoints = waypoints = [[-0.8,0], [-0.8,0.8], [0.8,0.8], [0.8,0]]

            self.intervals = np.linspace(30,150,20).astype(int)
            self.waypoints = [[-0.8,i] for i in np.linspace(0,0.9,5)] + [[i,0.9] for i in np.linspace(-0.8,0.8,10)] + [[0.8,i] for i in np.linspace(0.9,0,5)]
            self.current_waypoint_id = 0

        if self.env_id == 'simple_dual_waypoints':
            left_limits = ([-0.9,0],[-0.1,0.9])
            right_limits = ([0.1,0],[0.9,0.9])
            self.waypoints = []
            for i in range(2):
                self.waypoints.append(np.random.uniform(*left_limits) if np.random.randint(2) else np.random.uniform(*right_limits))
            print('waypoints', self.waypoints)
            self.intervals = [50,100]
            self.current_waypoint_id = 0

        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.05
        # physical damping
        self.damping = 0.05
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        ## wall positions
        self.wall_pos = [-1,1,-1,1] # (xmin, xmax) vertical and  (ymin,ymax) horizontal walls
        # number of steps that have been taken
        self.steps = 0
        self.max_steps_episode = 128
        self.leader_name = 0
        self.traj_points = []   # to draw a trajectory 
        

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]
    
    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # apply wall_collision force
        p_force = self.apply_wall_collision_force(p_force)

        # integrate physical state
        self.integrate_state(p_force)
        
        self.update_goal_state()
        
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        self.steps += 1

    def update_goal_state(self):
        if self.env_id in ['simple_waypoints', 'simple_dual_waypoints']:
            if self.steps<=self.intervals[-1] and self.steps == self.intervals[self.current_waypoint_id]:
                self.current_waypoint_id += 1
                num_waypoints = len(self.waypoints)
                self.landmarks[0].state.p_pos = self.waypoints[min(num_waypoints-1, self.current_waypoint_id)]
        
    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    ## apply wall collision force
    def apply_wall_collision_force(self, p_force):
        for a,agent in enumerate(self.agents):
            f = self.get_wall_collision_force(agent)
            if(f is not None):
                if(p_force[a] is None): p_force[a] = 0.0
                p_force[a] = f + p_force[a] 
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

     # collision force with wall
    def get_wall_collision_force(self, entity):
        if not entity.collide:
            return([None]) # not a collider
        
        xmin,xmax,ymin,ymax = self.wall_pos
        x,y = entity.state.p_pos
        size = entity.size
        dists = np.array([x-size-xmin, xmax-x-size, y-size-ymin, ymax-y-size])

        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -dists/k)*k
        fx1,fx2,fy1,fy2 = self.contact_force * penetration
        force = [fx1-fx2,fy1-fy2]
        return force

