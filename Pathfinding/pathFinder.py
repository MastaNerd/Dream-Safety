import cv2
import numpy as np
import networkx as nx
import json

def load_graph_and_image(gexf_file, original_image_path):
    img = cv2.imread(original_image_path)
    if img is None:
        raise FileNotFoundError(f"Original image not found at {original_image_path}")
    
    G = nx.read_gexf(gexf_file)
    
    img_copy = img.copy()
    
    pos = {}
    for node, data in G.nodes(data=True):
        pos[node] = (float(data['pos_x']), float(data['pos_y']))
    
    return G, pos, img_copy

def draw_graph(img, G, pos):
    for u, v, data in G.edges(data=True):
        if 'path' in data:
            try:
                path = json.loads(data['path'])
                pts = np.array([(int(x), int(y)) for x, y in path], dtype=np.int32)
                cv2.polylines(img, [pts], False, (0, 200, 255), 2)  # Orange edges
            except:
                pass
    
    for node, (x, y) in pos.items():
        cv2.circle(img, (int(x), int(y)), 8, (0, 255, 0), -1)  # Green nodes
        cv2.putText(img, node, (int(x)+10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

def find_and_draw_shortest_path(G, pos, img, node1, node2):
    try:
        path_nodes = nx.shortest_path(G, source=node1, target=node2, weight='weight')
        path_length = nx.shortest_path_length(G, source=node1, target=node2, weight='weight')
        
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                if 'path' in edge_data:
                    try:
                        pts = np.array([(int(x), int(y)) for x, y in json.loads(edge_data['path'])], dtype=np.int32)
                        cv2.polylines(img, [pts], False, (0, 0, 255), 4)
                    except:
                        pass
        
        cv2.circle(img, (int(pos[node1][0]), int(pos[node1][1])), 12, (0, 0, 255), -1)
        cv2.circle(img, (int(pos[node2][0]), int(pos[node2][1])), 12, (0, 0, 255), -1)
        
        last_edge = G.get_edge_data(path_nodes[-2], path_nodes[-1])
        if 'path' in last_edge:
            path_points = json.loads(last_edge['path'])
            mid_point = path_points[len(path_points)//2]
            cv2.putText(img, f"{path_length:.1f} px", (int(mid_point[0]), int(mid_point[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        print(f"Shortest path: {' -> '.join(path_nodes)}")
        print(f"Total distance: {path_length:.1f} pixels")
        
        return path_nodes
        
    except nx.NetworkXNoPath:
        print(f"No path exists between {node1} and {node2}")
        return None

def main():
    gexf_file = "/Users/vibhushsivakumar/Desktop/Dream Safety/Pathfinding/bmhsGraph.gexf"
    original_image_path = "/Users/vibhushsivakumar/Desktop/Dream Safety/Pathfinding/BMHS_FloorPlan.JPG"
    
    try:
        G, pos, img = load_graph_and_image(gexf_file, original_image_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    cv2.namedWindow("Floor Plan Pathfinder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Floor Plan Pathfinder", 1000, 800)
    
    draw_graph(img, G, pos)
    cv2.imshow("Floor Plan Pathfinder", img)
    
    selected_nodes = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_nodes, img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            closest_node = None
            min_dist = float('inf')
            
            for node, (nx_pos, ny_pos) in pos.items():
                dist = np.sqrt((x - nx_pos)**2 + (y - ny_pos)**2)
                if dist < min_dist and dist < 20:
                    min_dist = dist
                    closest_node = node
            
            if closest_node:
                selected_nodes.append(closest_node)
                print(f"Selected node: {closest_node}")
                
                img = cv2.imread(original_image_path)
                draw_graph(img, G, pos)
                
                for node in selected_nodes:
                    cv2.circle(img, (int(pos[node][0]), int(pos[node][1])), 
                              12, (255, 0, 0), -1)
                
                cv2.imshow("Floor Plan Pathfinder", img)
                
                if len(selected_nodes) == 2:
                    path = find_and_draw_shortest_path(G, pos, img, selected_nodes[0], selected_nodes[1])
                    cv2.imshow("Floor Plan Pathfinder", img)
                    selected_nodes = []
    
    cv2.setMouseCallback("Floor Plan Pathfinder", mouse_callback)
    
    print("Instructions:")
    print("1. Click on two nodes to find the shortest path between them")
    print("2. The path will be shown in red on your floor plan")
    print("3. Press 'r' to reset the view")
    print("4. Press 'q' to quit")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            img = cv2.imread(original_image_path)
            draw_graph(img, G, pos)
            cv2.imshow("Floor Plan Pathfinder", img)
            selected_nodes = []
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()