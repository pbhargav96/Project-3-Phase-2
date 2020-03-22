import numpy as np
from collections import deque
import math
import pygame
from pygame.locals import QUIT, MOUSEBUTTONUP
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import cv2

step=10

class Nodes:
    def __init__(self,state):
        self.state = state
        self.cost = math.inf
        self.parent = None
'''
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
        '''

# To check if the new point lies in the circle obstacle
def circle(position,crad):
    #crad = clearance + radius
    x = position[0]
    y = position[1]
    location = np.sqrt((x -225)**2+(y -150)**2)
    if location <=  (25+crad):
    	return True
    else:
    	return False

def ellipse(position,crad):
    #crad = clearance + radius
    x = position[0]
    y = position[1]
	#center = [150,100]
    location = ((float(x)-150)**2/(40+crad)**2) + ((float(y)-100)**2/(20+crad)**2)
    if location < 1 :
    	return True
    else:
        return False

def rhombus(position,crad):
    #crad = clearance + radius
    x = position[0]
    y = position[1]

    # mentioning the 4 co-ordinates
    pt1 = [225,10]
    pt2 = [200,25]
    pt3 = [225,40]
    pt4 = [250,25]

    m_pt1_pt2 = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    m_pt3_pt4 = (pt3[1]-pt4[1])/(pt3[0]-pt4[0])
    m_pt1_pt4 = (pt1[1]-pt4[1])/(pt1[0]-pt4[0])
    m_pt2_pt3 = (pt2[1]-pt3[1])/(pt2[0]-pt3[0])

    #solve for y intercept  y=mx+b or b = y-mx
    b_bottom = pt1[1]-m_pt1_pt2*pt1[0]
    b_top = pt3[1]-m_pt3_pt4*pt3[0]
    b_left = pt2[1]-m_pt2_pt3*pt2[0]
    b_right = pt4[1]-m_pt1_pt4*pt4[0]

    # 0 = y -mx -b
    top_line = y - m_pt3_pt4*x-(b_top+crad)
    bottom_line = y - m_pt1_pt2*x -(b_bottom-crad)
    left_line = y -m_pt2_pt3*x-(b_left+crad)
    right_line = y - m_pt1_pt4*x -(b_right-crad)

    if top_line <= 0 and bottom_line >= 0 and left_line <= 0 and right_line >= 0:
        return True
    else:
        return False

def rectangle(position,crad):
    #crad = clearance + radius
    x = position[0]
    y = position[1]

    # mentioning the 4 co-ordinates
    pt1 = [95,30]
    pt2 = [30.5,67.5]
    pt3 = [35.5,76.1]
    pt4 = [100,38.6]

    m_pt1_pt2 = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])  # Bottom line
    m_pt1_pt4 = (pt1[1]-pt4[1])/(pt1[0]-pt4[0])  # Right line
    m_pt3_pt4 = (pt3[1]-pt4[1])/(pt3[0]-pt4[0])  # Top line
    m_pt2_pt3 = (pt2[1]-pt3[1])/(pt2[0]-pt3[0])  # Left line

    b_bottom = pt1[1]-m_pt1_pt2*pt1[0]
    b_right = pt4[1]-m_pt1_pt4*pt4[0]
    b_top = pt3[1]-m_pt3_pt4*pt3[0]
    b_left = pt2[1]-m_pt2_pt3*pt2[0]

    bottom_line = y - m_pt1_pt2*x -(b_bottom-crad)
    right_line = y - m_pt1_pt4*x -(b_right-crad)
    top_line = y - m_pt3_pt4*x-(b_top+crad+1)
    left_line = y -m_pt2_pt3*x-(b_left+crad+2)

    if top_line <= 0 and bottom_line >= 0 and left_line <= 0 and right_line >= 0:
        return True
    else:
        return False



def polygon(position,crad):
    #crad = clearance + radius
    x = position[0]
    y = position[1]

    # mentioning the 6 co-ordinates
    pt1 = [20,120]
    pt2 = [50,150]
    pt3 = [75,120]
    pt4 = [100,150]
    pt5 = [75,185]
    pt6 = [25,185]


    m_pt2_pt5 = (pt2[1]-pt5[1])/(pt2[0]-pt5[0])
    m_pt1_pt2 = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    m_pt2_pt3 = (pt2[1]-pt3[1])/(pt2[0]-pt3[0])
    m_pt3_pt4 = (pt3[1]-pt4[1])/(pt3[0]-pt4[0])
    m_pt4_pt5 = (pt4[1]-pt5[1])/(pt4[0]-pt5[0])
    m_pt5_pt6 = (pt5[1]-pt6[1])/(pt5[0]-pt6[0])
    m_pt1_pt6 = (pt1[1]-pt6[1])/(pt1[0]-pt6[0])


    b_diag_1 = pt5[1]-m_pt2_pt5*pt5[0]
    b_line_1 = pt1[1]-m_pt1_pt2*pt1[0]
    b_line_2 = pt2[1]-m_pt2_pt3*pt2[0]
    b_line_3 = pt3[1]-m_pt3_pt4*pt3[0]
    b_line_4 = pt4[1]-m_pt4_pt5*pt4[0]
    b_line_5 = pt5[1]-m_pt5_pt6*pt5[0]
    b_line_6 = pt6[1]-m_pt1_pt6*pt6[0]


    diag_1 = y - m_pt2_pt5*x -(b_diag_1) # the interior diagonal joining pt 2 & 5
    line_1 = y - m_pt1_pt2*x -(b_line_1-crad)
    line_2 = y - m_pt2_pt3*x-(b_line_2+crad)
    line_3 = y - m_pt3_pt4*x-(b_line_3+crad)
    line_4 = y - m_pt4_pt5*x -(b_line_4-crad)
    line_5 = y - m_pt5_pt6*x -(b_line_5-crad)
    line_6 = y - m_pt5_pt6*x -(b_line_6-crad)



    if line_6 <= 0 and line_5 <= 0 and line_1>=0 and diag_1 >=0:
        return True

    if line_4 <= 0 and line_3 >= 0 and line_2 >= 0 and diag_1 <= 0:
        return True

    else:
        return False


def coll_check(position,crad):
    if rhombus(position,crad):
        return True
    elif circle(position,crad):
        return True
    elif ellipse(position,crad):
        return True


    elif rectangle(position,crad):
        return True
    elif polygon(position,crad):
        return True
    else:
        return False



def moveLeft30(position,crad):
    x = position[0]
    y = position[1]
    theta = position[2]
    theta1 = math.radians(theta+30)
    cost = 1
    new_x = round(2*(x+step*math.cos(theta1)))/2
    new_y = round(2*(y+step*math.sin(theta1)))/2
    new_theta = theta + 30
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[0],position[1]),crad):
        updated_pos=[new_x,new_y,new_theta]
        return updated_pos,cost
    else:
        return None,None

def moveLeft60(position,crad):
    x = position[0]
    y = position[1]
    theta = position[2]
    theta1 = math.radians(theta+60)
    cost = 1
    new_x = round(2*(x+step*math.cos(theta1)))/2
    new_y = round(2*(y+step*math.sin(theta1)))/2
    new_theta = theta + 60
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[0],position[1]),crad):
        updated_pos=[new_x,new_y,new_theta]
        return updated_pos,cost
    else:
        return None,None

def moveRight30(position,crad):
    x = position[0]
    y = position[1]
    theta = position[2]
    theta1 = math.radians(theta-30)
    cost = 1
    new_x = round(2*(x+step*math.cos(theta1)))/2
    new_y = round(2*(y+step*math.sin(theta1)))/2
    new_theta = theta- 30
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[0],position[1]),crad):
        updated_pos=[new_x,new_y,new_theta]
        return updated_pos,cost
    else:
        return None,None

def moveRight60(position,crad):
    x = position[0]
    y = position[1]
    theta=position[2]
    theta1 = math.radians(theta-60)
    cost = 1
    new_x = round(2*(x+step*math.cos(theta1)))/2
    new_y = round(2*(y+step*math.sin(theta1)))/2
    new_theta = theta-60
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[0],position[1]),crad):
        updated_pos=[new_x,new_y,new_theta]
        return updated_pos,cost
    else:
        return None,None

def moveStraight(position,crad):
    x = position[0]
    y = position[1]
    theta = position[2]
    theta1 = math.radians(theta)
    cost = 1
    new_x = round(2*(x+step*math.cos(theta1)))/2
    new_y = round(2*(y+step*math.sin(theta1)))/2
    new_theta = theta + 0
    if new_theta >= 360:
        new_theta = new_theta-360
    if new_theta <= -360:
        new_theta = new_theta+360
    if not coll_check((position[0],position[1]),crad):
        updated_pos=[new_x,new_y,new_theta]
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


def AStar_Algo(start,goal,crad,fig,ax):
    visited = []
    s_node = Nodes(start)
    s_node.cost = 0
    Q = [s_node]
    # imap[start[1], start[0]] = [255, 0, 0]
    # imap[goal[1], goal[0]] = [0, 0, 255]
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
                    return new_node.parent,new_node

                dist = math.inf
                for nodes in visited:
                    dist = min(dist, math.sqrt((n[0][0] - nodes[0])**2+(n[0][1] - nodes[1])**2))
                if dist > 2.5:
                # if n[0] not in visited:
                    new_node.cost = n[1]+new_node.parent.cost + CostToGoal(n[0],goal)
                    visited.append(new_node.state)
                    Q.append(new_node)

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


def plot_workspace(xi,yi,xg,yg):
    fig, ax = plt.subplots()


    verts2=[(95,200-170),(30.5,200-132.5),(35.5,200-123.9),(100,200-161.4),(95,200-170)]
    codes2 = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path2 = Path(verts2, codes2)
    patch2 = patches.PathPatch(path2, facecolor='blue', lw=0)
    ax.add_patch(patch2)

    verts3=[(225,200-190),(200,200-175),(225,200-160),(250,200-175),(225,200-190)]
    codes3 = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path3 = Path(verts3, codes3)
    patch3 = patches.PathPatch(path3, facecolor='blue', lw=0)
    ax.add_patch(patch3)

    verts4=[(20,200-80),(25,200-15),(75,200-15),(50,200-50),(20,200-80)]
    codes4 = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path4 = Path(verts4, codes4)
    patch4 = patches.PathPatch(path4, facecolor='blue', lw=0)
    ax.add_patch(patch4)

    verts5=[(50,200-50),(75,200-15),(100,200-50),(75,200-80),(50,200-50)]
    codes5 = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path5 = Path(verts5, codes5)
    patch5 = patches.PathPatch(path5, facecolor='blue', lw=0)
    ax.add_patch(patch5)

    ax.add_patch(patches.Circle((225, 200-50), radius=25, color='blue', lw=2))
    ax.add_patch(patches.Ellipse((150, 200-100), 80, 40, 0, color='blue', lw=2))

    ax.set_xlim(0, 300)
    ax.set_ylim(0, 200)
    return fig,ax


#
# class Robot:
#     def __init__(self, crad, int_pos, final_pos):
#         self.crad = crad
#         self.int_pos = int_pos
#         self.final_pos = final_pos


print("Enter the start node coordinates")
xi=int(input("x =  "))
yi=int(input("y =  "))
o_i=int(input("Orientation = "))
start=[xi,yi,o_i]
print("Enter the goal node coordinates")
xg=int(input("x =  "))
yg=int(input("y =  "))
goal=[xg,yg]
print("Enter the clearance")
cl = int(input("clearance =  "))
print("Enter the radius")
rad = int(input("radius =  "))
crader = cl + rad
#step = 10
#goal_thresh=3


if coll_check(goal,crader) or coll_check(start,crader):
    print("Start or goal nodes lie in the obstacle space")
    exit()
if start[0]<0 or start[0]>300 or start[1]<0 or start[1]>200:
    print("Start nodes beyond the workspace")
    exit()

if goal[0]<0 or goal[0]>300 or goal[1]<0 or goal[1]>200:
    print("goal nodes beyond the workspace")
    exit()

# robo = Robot(crader, start, goal)


    #start = [5,5]
    #goal = [100,5]
# img = np.ones((300, 200, 3), np.uint8)
# img1 = plot_workspace(img,crader)
# parent,image,vis = AStar_Algo(start,goal,img1,crader)
#n_list = track_back(parent)

fig,ax = plot_workspace(xi,yi,xg,yg)


parent, final_node = AStar_Algo(start,goal,crader,fig,ax)
plt.pause(1)

if parent is not None:
    parent_list = track_back(parent)
    xend = final_node.state[0]
    yend = final_node.state[1]
    first_parent = final_node.parent
    x = first_parent.state[0]
    y = first_parent.state[1]
    ax.arrow(x,y,xend-x,yend-y,length_includes_head=True,head_width=3, head_length=1,color="red")
    xend = x
    yend = y
    for parent in parent_list:
        x = parent.state[0]
        y = parent.state[1]
        # orientation = parent.orientation
        ax.arrow(x,y,xend-x,yend-y,length_includes_head=True,head_width=3, head_length=1,color="red")
        plt.draw()
        plt.pause(1)
        xend = x
        yend = y
    plt.pause(10)

else:
    print("No path to goal point")

# if parent is not None:
#     nodes_list = track_back(parent)
#     for elem in nodes_list:
#         x = elem.state[0]
#         y = elem.state[1]
#         #image[x, y] = [0, 255, 0]
#
#         resized_new = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#         cv2.imshow("Figure", resized_new)
#         print("abcd")
#
#         # cv2.imshow("Figure",plot_workspace(img1,crader))
#         cv2.waitKey(100)
# else:
#     print("Sorry, result could not be reached")
#
# print("Press any key to Quit")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# '''
#     pygame.init()
#     BLACK = [0, 0, 0]
#     WHITE = [255, 255, 255]
#     GREEN = [0, 255, 0]
#     RED = [255, 0, 0]
#     BLUE = [0,0,255]
#     VIOLET = [255,0,255]
#     size = [300*3, 200*3]
#     screen = pygame.display.set_mode(size)
#     pygame.display.set_caption("Visualization")
#     clock = pygame.time.Clock()
# '''
# '''
#     #screen.fill(WHITE)
#     #pygame.display.flip()
#     for i in imager:
#         pygame.time.wait(1)
#         pygame.draw.rect(screen, RED, [i[0]*3,i[1]*3,1,1])
#         pygame.display.flip()
#         for event in pygame.event.get():
#             if event.type == MOUSEBUTTONUP:
#                 None
# '''
# '''
#     pygame.draw.rect(screen, BLUE, [start[0]*3,start[1]*3,1,1])
#     pygame.draw.rect(screen, VIOLET, [goal[0]*3,goal[1]*3,1,1])
#     pygame.display.flip()
#     for v in vis:
#
#         pygame.time.wait(1)
#
#         pygame.draw.rect(screen, BLACK, [v[0]*3,v[1]*3,1,1])
#
#         pygame.display.flip()
#         for event in pygame.event.get():
#             if event.type == MOUSEBUTTONUP:
#                 None
#
#
#     for elem in n_list:
#         pygame.time.wait(1)
#         pygame.draw.rect(screen, GREEN, [elem.state[0]*3, elem.state[1]*3,1,1])
#         pygame.display.flip()
#         for event in pygame.event.get():
#             if event.type == MOUSEBUTTONUP:
#                 None
#
#     pygame.display.flip()
#     pygame.time.wait(1500)
#     done = False
#     while not done:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 done = True
#                 pygame.quit()
#                 break
#  '''
