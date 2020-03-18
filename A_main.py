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