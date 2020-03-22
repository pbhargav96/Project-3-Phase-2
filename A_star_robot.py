import numpy as np
from collections import deque
import math
import pygame
from pygame.locals import QUIT, MOUSEBUTTONUP
import matplotlib.pyplot as plt
import cv2

step=10

class Nodes:
    def __init__(self,state):
        self.state = state
        self.cost = math.inf
        self.parent = None


# To check if the new point lies in the circle obstacle
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

def moveLeft30(position,crad):
    x = position[1]
    y = position[0]
    theta = position[2]
    theta1 = math.radians(theta+30)
    cost = 1
    new_x = round(2*(x+step*math.sin(theta1)))/2
    new_y = round(2*(y+step*math.cos(theta1)))/2
    new_theta = theta + 30
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[1],position[0]),crad):
        updated_pos=[new_y,new_x,new_theta]
        return updated_pos,cost
    else:
        return None,None

def moveLeft60(position,crad):
    x = position[1]
    y = position[0]
    theta = position[2]
    theta1 = math.radians(theta+60)
    cost = 1
    new_x = round(2*(x+step*math.sin(theta1)))/2
    new_y = round(2*(y+step*math.cos(theta1)))/2
    new_theta = theta + 60
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[1],position[0]),crad):
        updated_pos=[new_y,new_x,new_theta]
        return updated_pos,cost
    else:
        return None,None

def moveRight30(position,crad):
    x = position[1]
    y = position[0]
    theta = position[2]
    theta1 = math.radians(theta-30)
    cost = 1
    new_x = round(2*(x+step*math.sin(theta1)))/2
    new_y = round(2*(y+step*math.cos(theta1)))/2
    new_theta = theta- 30
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[1],position[0]),crad):
        updated_pos=[new_y,new_x,new_theta]
        return updated_pos,cost
    else:
        return None,None

def moveRight60(position,crad):
    x = position[1]
    y = position[0]
    theta=position[2]
    theta1 = math.radians(theta-60)
    cost = 1
    new_x = round(2*(x+step*math.sin(theta1)))/2
    new_y = round(2*(y+step*math.cos(theta1)))/2
    new_theta = theta-60
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[1],position[0]),crad):
        updated_pos=[new_y,new_x,new_theta]
        return updated_pos,cost
    else:
        return None,None

def moveStraight(position,crad):
    x = position[1]
    y = position[0]
    theta = position[2]
    theta1 = math.radians(theta)
    cost = 1
    new_x = round(2*(x+step*math.sin(theta1)))/2
    new_y = round(2*(y+step*math.cos(theta1)))/2
    new_theta = theta + 0
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[1],position[0]),crad):
        updated_pos=[new_y,new_x,new_theta]
        return updated_pos,cost
    else:
        return None,None



def mover(position,action,crad):
    if action == 'left30':
        [updated_pos,cost] = moveLeft30(position,crad)
    elif action == 'left60':
        [updated_pos,cost] = moveLeft60(position,crad)
    elif action == 'straight':
        [updated_pos,cost] = moveStraight(position,crad)
    elif action == 'right30':
        [updated_pos,cost] = moveRight30(position,crad)
    elif action == 'right60':
        [updated_pos,cost] = moveRight60(position,crad)
    return updated_pos,cost

def mappingofneighbors(current_position,crad):
    neighbors=['left30','left60','straight','right30','right60']

    active_points=[]
    for action in neighbors:
        new_pos,cost = mover(current_position,action,crad)
        active_points.append((new_pos,cost))
    #active_points = [new_pos for new_pos in active_points if new_pos!= None]
    return active_points

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

def CostToGoal(point,goal):
    xp = point[0]
    yp = point[1]
    xg = goal[0]
    yg = goal[1]
    euclidean_distance= np.sqrt((xp-xg)**2 + (yp-yg)**2)
    return euclidean_distance


def AStar_Algo(start,goal,img,crad):
    visited = []
    s_node = Nodes(start)
    s_node.cost = 0
    Q = [s_node]
    img[start[1], start[0]] = [255, 0, 0]
    img[goal[1], goal[0]] = [0, 0, 255]
    while Q:
        current = priority_pop(Q)
        visited.append(current.state)
        neighbors = mappingofneighbors(current.state,crad)
        for n in neighbors:
            if n[0] is not None:
                new_node = Nodes(n[0])
                # print(n[0])
                new_node.parent = current
                # if n[0][0]==goal[0] and n[0][1]==goal[1]:
                if math.sqrt((n[0][0] - goal[0])**2+(n[0][1] - goal[1])**2) <= 3:
                    print("Goal Reached")
                    return new_node,img,visited

                #To increase speed of simualtion, setting a threshold value to limit the nodes considered
                dist = math.inf
                for nodes in visited:
                    dist = min(dist, math.sqrt((n[0][0] - nodes[0])**2+(n[0][1] - nodes[1])**2))
                if dist > 2.5:
                # if n[0] not in visited:
                    new_node.cost = n[1]+new_node.parent.cost + CostToGoal(n[0],goal)
                    visited.append(new_node.state)
                    Q.append(new_node)
#                     print(len(visited))
                else:
                    node_id = find_node(new_node,Q)

                    if node_id:
                        t_node = queue[node_id]
                        if t_node.cost> n[1]+new_node.parent.cost + CostToGoal(n[0],goal):
                            t_node.cost= n[1]+new_node.parent.cost + CostToGoal(n[0],goal)
                            t_node.parent = current
            else:
                continue

    return None,None,None


#
# class Robot:
#     def __init__(self, crad, int_pos, final_pos):
#         self.crad = crad
#         self.int_pos = int_pos
#         self.final_pos = final_pos

def main():
    print("Enter the start node coordinates")
    xi=int(input("x =  "))
    yi=int(input("y =  "))
    o_i=int(input("Orientation = "))
    start=[xi,200-yi,o_i]
    print("Enter the goal node coordinates")
    xg=int(input("x =  "))
    yg=int(input("y =  "))
    goal=[xg,200-yg]
    print("Enter the clearance")
    cl = int(input("clearance =  "))
    print("Enter the radius")
    rad = int(input("radius =  "))
    crader = cl + rad
    step = 10

    #To check if the goal and start are in the obstacle space and if they are viable points
    if coll_check(goal,crader) or coll_check(start,crader):
        print("Start or goal nodes lie in the obstacle space")
        exit()
    if start[0]<0 or start[0]>300 or start[1]<0 or start[1]>200:
        print("Start nodes beyond the workspace")
        exit()

    if goal[0]<0 or goal[0]>300 or goal[1]<0 or goal[1]>200:
        print("goal nodes beyond the workspace")
        exit()

    img = np.zeros((201,301,3), np.uint8)
    imager = []

    BLACK = (0,0,0)
    WHITE = (255,255,255)
    #crader = 10

    for i in range(201):
        for j in range(301):
            if coll_check([i,j],crader):
                img[i][j] = BLACK
            else:
                img[i][j] = WHITE
    for i in range(201):
        for j in range(301):
            if coll_check([i,j],crader):
                imager.append([j,i])


    #start = [5,5]
    #goal = [100,5]
    parent,image,vis = AStar_Algo(start,goal,img,crader)

    n_list = track_back(parent)

    pygame.init()
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    GREEN = [0, 255, 0]
    RED = [255, 0, 0]
    BLUE = [0,255,255]
    VIOLET = [255,0,255]
    size = [300*3, 200*3]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Visualization")
    clock = pygame.time.Clock()

    screen.fill(BLACK)
    pygame.display.flip()
    for i in imager:
        pygame.time.wait(1)
        pygame.draw.rect(screen, RED, [i[0]*3,i[1]*3,0,0])
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                None

    pygame.draw.rect(screen, BLUE, [start[0]*3,start[1]*3,-2,-2])
    pygame.draw.rect(screen, VIOLET, [goal[0]*3,goal[1]*3,-2,-2])
    pygame.display.flip()
    for v in vis:

        pygame.time.wait(1)

        pygame.draw.rect(screen, GREEN, [v[0]*3,v[1]*3,1,1])

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                None


    for elem in n_list:
        pygame.time.wait(1)
        pygame.draw.rect(screen, WHITE, [elem.state[0]*3, elem.state[1]*3,-1,-1])
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONUP:
                None

    pygame.display.flip()
    pygame.time.wait(1500)
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                break


    #cv2.imshow('Environment',img)
    #cv2.waitKey(0)
if __name__ == '__main__':
    main()
