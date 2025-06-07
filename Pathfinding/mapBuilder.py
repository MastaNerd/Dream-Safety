import cv2
import numpy as np
import networkx as nx
import json

img_path = "/Users/vibhushsivakumar/Desktop/Dream Safety/Pathfinding/BMHS_FloorPlan.JPG"
img = cv2.imread(img_path)
if img is None:
    print("Error: Image not found!")
    exit()

img_copy = img.copy()
G = nx.Graph()
drawing = False
current_path = []
nodes = {}

_, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

# --- Node Placement ---
def add_node(event, x, y, flags, param):
    global img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        node_id = f"Node_{len(nodes)+1}"
        nodes[node_id] = (x, y)
        cv2.circle(img_copy, (x, y), 8, (0, 255, 0), -1)  # Green dot
        cv2.putText(img_copy, node_id, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        G.add_node(node_id, pos_x=float(x), pos_y=float(y))
        cv2.imshow("Map", img_copy)

def draw_edge(event, x, y, flags, param):
    global drawing, current_path, img_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_path = [(x, y)]
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)  # Red start dot
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and binary[y, x] == 255:
            cv2.line(img_copy, current_path[-1], (x, y), (255, 0, 0), 2)
            current_path.append((x, y))
            cv2.imshow("Map", img_copy)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(current_path) >= 2:
            start_node = min(nodes.items(), key=lambda n: np.linalg.norm(np.array(n[1]) - np.array(current_path[0])))[0]
            end_node = min(nodes.items(), key=lambda n: np.linalg.norm(np.array(n[1]) - np.array(current_path[-1])))[0]
            
            distance = sum(np.linalg.norm(np.array(current_path[i]) - np.array(current_path[i-1])) 
                         for i in range(1, len(current_path)))
            
            path_str = json.dumps(current_path)
            G.add_edge(start_node, end_node, 
                      weight=round(distance, 1),
                      path=path_str)
            
            print(f"Added edge: {start_node} -> {end_node} | Distance: {distance:.1f} px")
            
            cv2.polylines(img_copy, [np.array(current_path)], False, (0, 255, 255), 2)
            mid_point = current_path[len(current_path)//2]
            cv2.putText(img_copy, f"{distance:.1f}", mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("Map", img_copy)
        current_path = []

def redraw_edges():
    global img_copy
    img_copy = img.copy()
    for node_id, (x, y) in nodes.items():
        cv2.circle(img_copy, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(img_copy, node_id, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    for u, v, data in G.edges(data=True):
        if 'path' in data:
            try:
                path = json.loads(data['path'])
                cv2.polylines(img_copy, [np.array(path)], False, (0, 255, 255), 2)
                mid_point = path[len(path)//2]
                cv2.putText(img_copy, f"{data['weight']:.1f}", mid_point, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except (json.JSONDecodeError, KeyError):
                pass

cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Map", 1000, 800)

print("STEP 1: Click to place nodes (rooms/hallways). Press 'n' when done.")
cv2.setMouseCallback("Map", add_node)
while True:
    cv2.imshow("Map", img_copy)
    if cv2.waitKey(1) & 0xFF == ord('n'):
        break

print("STEP 2: Click+drag to draw paths around walls. Press 's' to save, 'r' to refresh.")
cv2.setMouseCallback("Map", draw_edge)
while True:
    cv2.imshow("Map", img_copy)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        G_save = nx.Graph()
        
        for node, data in G.nodes(data=True):
            G_save.add_node(node, pos_x=str(data['pos_x']), pos_y=str(data['pos_y']))
            
        for u, v, data in G.edges(data=True):
            G_save.add_edge(u, v, 
                          weight=str(data['weight']),
                          path=data['path']) 
        
        nx.write_gexf(G, "/Users/vibhushsivakumar/Desktop/Dream Safety/Pathfinding/bmhsGraph.gexf")
        print("Full graph saved to 'bmhsGraph.gexf'")
        
    elif key == ord('r'):
        redraw_edges()
        cv2.imshow("Map", img_copy)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()