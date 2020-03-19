import numpy as np

class Nodes:
    def __init__(self,state):
        self.state = state
        self.cost = 19999
        self.parent = None

def coll_circle(position,crad):
	p_x = position[1]
	p_y = position[0]
	location = np.sqrt((p_x -225)**2+(p_y -50)**2) - (25+crad)
	if location < 0:
		return True
	else:
		return False

def coll_ellipse(position,crad):
	p_x = position[1]
	p_y = position[0]
	center = [150,100]
	location = ((float(p_x)-150)**2/(40+crad)**2) + ((float(p_y)-100)**2/(20+crad)**2) -1.0
	if location < 0:
		return True
	else:
		return False

def coll_rhom(position,crad):
	p_x = position[1]
	p_y = position[0]
	m_1,c_1 = -0.6,295.0
	m_2,c_2 = 0.6,55.0
	m_3,c_3 = -0.6,325.0
	m_4,c_4 = 0.6,25.0

	if p_y - p_x*m_1>(c_1-crad*math.cos(math.atan(m_1))) and p_y - p_x*m_2<(c_2+crad*math.cos(math.atan(m_2)))\
    and p_y - p_x*m_3<(c_3+crad*math.cos(math.atan(m_3))) and p_y - p_x*m_4>(c_4-crad*math.cos(math.atan(m_4))):
		return True
	else:
		return False

def coll_rect(position,crad):
	p_x = position[1]
	p_y = position[0]
	m_1,c_1 = -1.6,322.0
	m_2,c_2 = 0.578125,104.1875
	m_3,c_3 = -1.6,182.6
	m_4,c_4 = 0.578125,115.078125

	if p_y - p_x*m_1<(c_1+crad*math.cos(math.atan(m_1))) and p_y - p_x*m_2>(c_2-crad*math.cos(math.atan(m_2))) and p_y - p_x*m_3>(c_3-crad*math.cos(math.atan(m_3)))\
    and p_y - p_x*m_4<(c_4+crad*math.cos(math.atan(m_4))):
		return True
	else:
		return False

def coll_poly1(position,crad):
	p_x = position[1]
	p_y = position[0]
	m_1,c_1 = -1.0,100.0
	m_2,c_2 = 1.2,-10.0
	m_3,c_3 = -1.2,170.0
	m_4,c_4 = 1.4,-90.0
	m_5,c_5 = 0.0,15.0
	m_6,c_6 = -13.0,340.0

	if p_y - p_x*m_1>(c_1+crad*math.cos(math.atan(m_1))) and p_y - p_x*m_2<(c_2+crad*math.cos(math.atan(m_2)))\
    and p_y - p_x*m_3<(c_3+crad*math.cos(math.atan(m_3))) and p_y - p_x*m_4>(c_4-crad*math.cos(math.atan(m_4))) and p_y - p_x*m_5>(c_5-crad*math.cos(math.atan(m_5)))\
    and p_y - p_x*m_6>(c_6-crad*math.cos(math.atan(m_6))):
		return True
	else:
		return False

def coll_poly2(position,crad):
	p_x = position[1]
	p_y = position[0]
	m_1,c_1 = -1.0,100.0
	m_2,c_2 = 1.2,-10.0
	m_3,c_3 = -1.2,170.0
	m_4,c_4 = 1.4,-90.0
	m_5,c_5 = 0.0,15.0
	m_6,c_6 = -13.0,340.0

	if p_y - p_x*m_1<=(c_1+crad*math.cos(math.atan(m_1))) and p_y - p_x*m_2<(c_2+crad*math.cos(math.atan(m_2)))\
    and p_y - p_x*m_3<(c_3-crad*math.cos(math.atan(m_3))) and p_y - p_x*m_4>(c_4-crad*math.cos(math.atan(m_4)))\
    and p_y - p_x*m_5>(c_5-crad*math.cos(math.atan(m_5))) and p_y - p_x*m_6>(c_6-crad*math.cos(math.atan(m_6))):
		return True
	else:
		return False

def coll_poly3(position,crad):
	p_x = position[1]
	p_y = position[0]
	m_1,c_1 = -1.0,100.0
	m_2,c_2 = 1.2,-10.0
	m_3,c_3 = -1.2,170.0
	m_4,c_4 = 1.4,-90.0
	m_5,c_5 = 0.0,15.0
	m_6,c_6 = -13.0,340.0

	if p_y - p_x*m_1<=(c_1+crad*math.cos(math.atan(m_1))) and p_y - p_x*m_2>=(c_2+crad*math.cos(math.atan(m_2)))\
    and p_y - p_x*m_3<(c_3-crad*math.cos(math.atan(m_3))) and p_y - p_x*m_4>(c_4+crad*math.cos(math.atan(m_4)))\
    and p_y - p_x*m_5>(c_5-crad*math.cos(math.atan(m_5))) and p_y - p_x*m_6>(c_6+crad*math.cos(math.atan(m_6))):
		return True
	else:
		return False

def coll_check(position,crad):
	if coll_rhom(position,crad):
		return True
	elif coll_circle(position,crad):
		return True
	elif coll_ellipse(position,crad):
		return True
	elif coll_rect(position,crad):
		return True
	elif coll_poly1(position,crad):
		return True
	elif coll_poly2(position,crad):
		return True
	elif coll_poly3(position,crad):
		return True
	else:
		return False
	
#Update A Star Algorithm from dijkstra and path generation
def priority_pop(queue):  # Priority Queue, outputs the node with least cost attached to it
    min_a = 0
    for elemt in range(len(queue)):
        if queue[elemt].cost < queue[min_a].cost:
            min_a = elemt
    return queue.pop(min_a)

def find_node(point, queue):
    for elem in queue:
        if elem.state == point:
            return queue.index(elem)
        else:
            return None

def track_back(node):
    print("Tracking Back")
    p = []
    p.append(node.parent)
    parent = node.parent
    if parent is None:
        return p
    while parent is not None:
        p.append(parent)
        parent = parent.parent
    return p

def color_pixel(image_color, point):
    #print(point)
    image_color[point[1], point[0]] = [255,0,255]
    return image_color

def CostToGoal(point,goal):
    xp = point[0]
    yp = point[1]
    xg = goal[0]
    yg = goal[1]
    euclidean_distance= np.sqrt((xp-xg)**2 + (yp-yg)**2)
    return euclidean_distance
def AStar_Algo(start,goal,imap,crad):
	visited = []
	s_node = Nodes(start)
	s_node.cost = 0
	Q = [s_node]
	imap[start[1], start[0]] = [255, 0, 0]
	imap[goal[1], goal[0]] = [0, 0, 255]
	while Q:
		current = priority_pop(Q)
		visited.append(current.state)
		neighbors = mappingofneighbors(current.state,crad)

		for n in neighbors:
			if n[0] is not None:

				new_node = Nodes(n[0])
				#print(n[0])
				new_node.parent = current

				if n[0][0]==goal[0] and n[0][1]==goal[1]:
					print("Goal Reached")
					return new_node,imap,visited

				if n[0] not in visited:
					new_node.cost = n[1]+new_node.parent.cost+CostToGoal(n[0],goal)
					visited.append(new_node.state)
					Q.append(new_node)
				else:
					node_id = find_node(new_node,Q)

					if node_id:
						t_node = queue[node_id]
						if t_node.cost> n[1]+new_node.parent.cost+CostToGoal(n[0],goal):
							t_node.cost= n[1]+new_node.parent.cost+CostToGoal(n[0],goal)
							t_node.parent = current
			else:
				continue

	return None,None,None
