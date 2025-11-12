#!/usr/bin/env python3
"""
Visualize Existing GEXF Nodes on Floor Plan
This script shows all the nodes you've already created in your GEXF file
"""

import cv2
import networkx as nx
import numpy as np

def visualize_gexf_nodes():
    """Load and visualize your existing GEXF file nodes"""
    
    print("🗺️  Visualizing Your Existing BMHS Navigation Nodes")
    print("=" * 50)
    
    try:
        # Load your existing files
        G = nx.read_gexf("BMHS_FloorPlan_combined.gexf")
        floor_plan = cv2.imread("BMHS_FloorPlan.JPG")
        
        if floor_plan is None:
            print("❌ Could not load BMHS_FloorPlan.JPG")
            return
        
        print(f"✅ Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # Extract node positions and floor info
    pos = {}
    floor_info = {}
    
    for node, data in G.nodes(data=True):
        pos[node] = (float(data['pos_x']), float(data['pos_y']))
        floor_info[node] = data.get('floor', 'Ground')
    
    # Create visualization
    img = floor_plan.copy()
    
    # Draw all nodes with different colors by floor
    floor_colors = {
        'Floor1': (100, 100, 255),  # Light blue
        'Floor2': (255, 100, 100),  # Light red
        'Ground': (150, 150, 150)   # Gray
    }
    
    floor_counts = {}
    
    for node, (x, y) in pos.items():
        floor = floor_info[node]
        color = floor_colors.get(floor, (150, 150, 150))
        
        # Count nodes per floor
        floor_counts[floor] = floor_counts.get(floor, 0) + 1
        
        # Draw node
        cv2.circle(img, (int(x), int(y)), 5, color, -1)
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 0), 1)  # Black border
        
        # Add node number for identification
        node_num = node.split('_')[-1] if '_' in node else node[-2:]
        cv2.putText(img, node_num, (int(x) + 7, int(y) - 7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    # Draw edges
    edge_count = 0
    stair_count = 0
    
    for edge in G.edges():
        node1, node2 = edge
        if node1 in pos and node2 in pos:
            pos1 = pos[node1]
            pos2 = pos[node2]
            
            # Check if it's a stair connection
            edge_data = G.get_edge_data(node1, node2)
            if edge_data and edge_data.get('is_stair') == 'true':
                cv2.line(img, (int(pos1[0]), int(pos1[1])), 
                        (int(pos2[0]), int(pos2[1])), (255, 0, 255), 2)  # Magenta for stairs
                stair_count += 1
            else:
                cv2.line(img, (int(pos1[0]), int(pos1[1])), 
                        (int(pos2[0]), int(pos2[1])), (200, 200, 200), 1)  # Gray for regular
            edge_count += 1
    
    # Add comprehensive legend
    legend_x, legend_y = 10, 30
    cv2.rectangle(img, (legend_x - 5, legend_y - 20), (legend_x + 200, legend_y + 120), (255, 255, 255), -1)
    cv2.rectangle(img, (legend_x - 5, legend_y - 20), (legend_x + 200, legend_y + 120), (0, 0, 0), 2)
    
    cv2.putText(img, "Your BMHS Navigation Graph", (legend_x, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    y_offset = 20
    for floor, color in floor_colors.items():
        count = floor_counts.get(floor, 0)
        if count > 0:
            cv2.circle(img, (legend_x + 10, legend_y + y_offset), 5, color, -1)
            cv2.putText(img, f"{floor}: {count} nodes", (legend_x + 25, legend_y + y_offset + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_offset += 15
    
    cv2.line(img, (legend_x + 5, legend_y + y_offset), (legend_x + 20, legend_y + y_offset), (200, 200, 200), 1)
    cv2.putText(img, f"Connections: {edge_count}", (legend_x + 25, legend_y + y_offset + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    y_offset += 15
    
    if stair_count > 0:
        cv2.line(img, (legend_x + 5, legend_y + y_offset), (legend_x + 20, legend_y + y_offset), (255, 0, 255), 2)
        cv2.putText(img, f"Stairs: {stair_count}", (legend_x + 25, legend_y + y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Save the visualization
    output_path = "Your_BMHS_Nodes_Visualization.jpg"
    cv2.imwrite(output_path, img)
    
    # Print statistics
    print(f"\n📊 Your Navigation Graph Statistics:")
    print(f"   Total Nodes: {len(G.nodes)}")
    print(f"   Total Edges: {len(G.edges)}")
    for floor, count in floor_counts.items():
        print(f"   {floor} Nodes: {count}")
    print(f"   Stair Connections: {stair_count}")
    print(f"   Regular Connections: {edge_count - stair_count}")
    
    # Check connectivity
    if nx.is_connected(G):
        print("   ✅ Graph is fully connected")
    else:
        components = list(nx.connected_components(G))
        print(f"   ⚠️  Graph has {len(components)} disconnected components")
        for i, component in enumerate(components, 1):
            print(f"      Component {i}: {len(component)} nodes")
    
    print(f"\n✅ Visualization saved as: {output_path}")
    print("   This shows all your existing nodes with their connections!")
    
    # Show some sample nodes for reference
    print(f"\n📍 Sample Nodes from Your GEXF:")
    sample_nodes = list(G.nodes())[:5]
    for node in sample_nodes:
        data = G.nodes[node]
        x, y = pos[node]
        floor = floor_info[node]
        print(f"   {node}: ({x:.0f}, {y:.0f}) on {floor}")
    
    return img, G, pos

if __name__ == "__main__":
    visualize_gexf_nodes()
