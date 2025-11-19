#!/usr/bin/env python3
"""
Build Enhanced Navigation Map for Dream Safety
This script processes your BMHS floor plan and creates an enhanced navigation graph
"""

import cv2
import numpy as np
from enhanced_pathfinding import FloorPlanProcessor, EnhancedPathfinder

def build_bmhs_navigation_map():
    """Build enhanced navigation map for BMHS floor plan"""
    
    print("🏫 Building Enhanced BMHS Navigation Map")
    print("=" * 50)
    
    # Initialize with your existing floor plan
    floor_plan_path = "BMHS_FloorPlan.JPG"
    
    try:
        processor = FloorPlanProcessor(floor_plan_path, pixels_per_meter=8.0)
        print(f"✅ Loaded floor plan: {floor_plan_path}")
    except FileNotFoundError:
        print(f"❌ Could not find floor plan: {floor_plan_path}")
        return None
    
    # Step 1: Preprocess the floor plan
    print("\n🔄 Step 1: Preprocessing floor plan...")
    processor.preprocess_floor_plan(threshold_method='adaptive')
    
    # Step 2: Create navigation nodes
    print("\n🔄 Step 2: Creating navigation grid...")
    nodes = processor.create_node_grid(
        grid_spacing=20,  # Node every 20 pixels
        min_clearance=5   # At least 5 pixels from walls
    )
    
    # Step 3: Connect nodes
    print("\n🔄 Step 3: Connecting navigation nodes...")
    connections = processor.connect_nodes(max_connection_distance=30)
    
    # Step 4: Add BMHS-specific locations
    print("\n🔄 Step 4: Adding BMHS locations...")
    
    # These coordinates should match your actual floor plan
    # You may need to adjust these based on your specific BMHS_FloorPlan.JPG
    bmhs_locations = {
        "Main Entrance": (150, 300),
        "Cafeteria": (400, 200),
        "Gymnasium": (600, 400),
        "Library": (300, 150),
        "Safe Room 1": (100, 100),
        "Safe Room 2": (700, 100),
        "Office": (200, 250),
        "Hallway A": (350, 300),
        "Hallway B": (450, 350),
        "Stairwell North": (250, 100),
        "Stairwell South": (550, 500),
        "Emergency Exit 1": (50, 200),
        "Emergency Exit 2": (750, 300),
        "Nurse Office": (180, 180),
        "Principal Office": (220, 200)
    }
    
    processor.add_special_locations(bmhs_locations)
    
    # Step 5: Save enhanced graph
    print("\n🔄 Step 5: Saving enhanced navigation graph...")
    output_gexf = "BMHS_Enhanced_Navigation.gexf"
    processor.save_graph(output_gexf)
    
    # Step 6: Create visualization
    print("\n🔄 Step 6: Creating visualization...")
    processor.visualize_graph(
        save_path="BMHS_Navigation_Visualization.png",
        show_connections=True
    )
    
    # Step 7: Test pathfinding capabilities
    print("\n🔄 Step 7: Testing pathfinding...")
    pathfinder = EnhancedPathfinder(processor.graph, processor.node_positions)
    
    # Test multiple pathfinding scenarios
    test_scenarios = [
        ("LOC_MAIN_ENTRANCE", "LOC_SAFE_ROOM_1", "Emergency Response"),
        ("LOC_CAFETERIA", "LOC_EMERGENCY_EXIT_1", "Evacuation Route"),
        ("LOC_OFFICE", "LOC_GYMNASIUM", "Officer Patrol"),
        ("LOC_LIBRARY", "LOC_SAFE_ROOM_2", "Student Evacuation")
    ]
    
    print("\n📍 Pathfinding Test Results:")
    print("-" * 40)
    
    for start, end, scenario in test_scenarios:
        result = pathfinder.find_optimal_path(start, end, algorithm='astar')
        
        if result['status'] == 'success':
            print(f"\n✅ {scenario}:")
            print(f"   Route: {start} → {end}")
            print(f"   Distance: {result['length_meters']:.1f} meters")
            print(f"   Nodes: {result['node_count']}")
            print(f"   Algorithm: {result['algorithm']}")
        else:
            print(f"\n❌ {scenario}: {result.get('error', 'Unknown error')}")
    
    # Generate summary statistics
    print(f"\n📊 Navigation Graph Statistics:")
    print(f"   Total Nodes: {len(processor.graph.nodes)}")
    print(f"   Total Connections: {len(processor.graph.edges)}")
    print(f"   Special Locations: {len(bmhs_locations)}")
    print(f"   Graph Density: {nx.density(processor.graph):.3f}")
    
    # Check graph connectivity
    if nx.is_connected(processor.graph):
        print("   ✅ Graph is fully connected")
    else:
        components = list(nx.connected_components(processor.graph))
        print(f"   ⚠️  Graph has {len(components)} disconnected components")
    
    print(f"\n🎉 Enhanced navigation map completed!")
    print(f"   Graph saved as: {output_gexf}")
    print(f"   Visualization saved as: BMHS_Navigation_Visualization.png")
    
    return processor, pathfinder

def create_integration_example():
    """Create example showing how to integrate with existing system"""
    
    integration_code = '''
# Integration with existing integrated.py
# Replace your existing PathfindingService.__init__ with:

def __init__(self, gexf_file, floor_plan_image):
    """Initialize enhanced pathfinding service"""
    # Load the enhanced graph
    self.G = nx.read_gexf(gexf_file)
    self.floor_plan = cv2.imread(floor_plan_image)
    
    # Extract enhanced node information
    self.pos = {}
    self.floor_info = {}
    self.special_locations = {}
    
    for node, data in self.G.nodes(data=True):
        self.pos[node] = (float(data['pos_x']), float(data['pos_y']))
        
        # Handle floor information (if available)
        if 'floor' in data:
            self.floor_info[node] = data['floor']
        else:
            self.floor_info[node] = "Ground"  # Default
        
        # Track special locations
        if data.get('is_special', False):
            location_type = data.get('location_type', 'Unknown')
            self.special_locations[location_type] = node
    
    # Enhanced location mapping
    self.location_nodes = self.special_locations
    
    # Officer positions (can be dynamically updated)
    self.officer_positions = {
        "Officer1": self.special_locations.get("Main Entrance", "LOC_MAIN_ENTRANCE"),
        "Officer2": self.special_locations.get("Hallway A", "LOC_HALLWAY_A"), 
        "Officer3": self.special_locations.get("Stairwell North", "LOC_STAIRWELL_NORTH")
    }
    
    print(f"Enhanced pathfinding initialized with {len(self.G.nodes)} nodes")
    print(f"Special locations: {list(self.special_locations.keys())}")
'''
    
    with open("integration_example.py", "w") as f:
        f.write(integration_code)
    
    print("📝 Integration example saved as: integration_example.py")

if __name__ == "__main__":
    import networkx as nx
    
    # Build the enhanced navigation map
    processor, pathfinder = build_bmhs_navigation_map()
    
    # Create integration example
    create_integration_example()
    
    print("\n🚀 Next Steps:")
    print("1. Review the generated visualization to verify node placement")
    print("2. Adjust location coordinates in bmhs_locations if needed")
    print("3. Replace your existing GEXF file with BMHS_Enhanced_Navigation.gexf")
    print("4. Update your integrated.py using the integration example")
    print("5. Restart your system to use the enhanced navigation!")
