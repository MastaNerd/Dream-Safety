#!/usr/bin/env python3
"""
Enhanced Dream Safety Navigation System
Converts floor plans to navigable graphs and provides intelligent pathfinding
"""

import cv2
import numpy as np
import networkx as nx
import json
import base64
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import euclidean
from skimage import morphology, measure
import matplotlib.pyplot as plt
from pathlib import Path

class FloorPlanProcessor:
    """Converts floor plan images into navigable node graphs"""
    
    def __init__(self, floor_plan_path: str, pixels_per_meter: float = 10.0):
        """
        Initialize the floor plan processor
        
        Args:
            floor_plan_path: Path to the floor plan image
            pixels_per_meter: Scale factor (pixels per real-world meter)
        """
        self.floor_plan_path = floor_plan_path
        self.pixels_per_meter = pixels_per_meter
        self.original_image = cv2.imread(floor_plan_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Could not load floor plan: {floor_plan_path}")
        
        self.processed_image = None
        self.walkable_mask = None
        self.graph = nx.Graph()
        self.node_positions = {}
        
    def preprocess_floor_plan(self, threshold_method='adaptive'):
        """
        Step 1: Convert floor plan to binary walkable/non-walkable mask
        
        Args:
            threshold_method: 'adaptive', 'otsu', or 'manual'
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply different thresholding methods
        if threshold_method == 'adaptive':
            # Adaptive thresholding works well for varying lighting
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        elif threshold_method == 'otsu':
            # Otsu's method automatically finds optimal threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Manual threshold (adjust based on your floor plan)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Clean up the binary image
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert if needed (white = walkable, black = walls)
        # Adjust this based on your floor plan colors
        if np.mean(binary) < 127:  # If mostly black, invert
            binary = cv2.bitwise_not(binary)
        
        self.walkable_mask = binary
        self.processed_image = binary
        
        print(f"✅ Floor plan preprocessed: {np.sum(binary == 255)} walkable pixels")
        return binary
    
    def create_node_grid(self, grid_spacing: int = 20, min_clearance: int = 10):
        """
        Step 2: Create a grid of nodes in walkable areas
        
        Args:
            grid_spacing: Distance between grid nodes (pixels)
            min_clearance: Minimum distance from walls (pixels)
        """
        if self.walkable_mask is None:
            raise ValueError("Must preprocess floor plan first")
        
        height, width = self.walkable_mask.shape
        nodes = []
        node_id = 0
        
        # Create distance transform to find areas far from walls
        dist_transform = cv2.distanceTransform(
            self.walkable_mask, cv2.DIST_L2, 5
        )
        
        # Sample grid points
        for y in range(grid_spacing // 2, height, grid_spacing):
            for x in range(grid_spacing // 2, width, grid_spacing):
                # Check if this point is walkable and has sufficient clearance
                if (self.walkable_mask[y, x] == 255 and 
                    dist_transform[y, x] >= min_clearance):
                    
                    node_name = f"N{node_id:04d}"
                    self.graph.add_node(node_name, 
                                      pos_x=x, pos_y=y, 
                                      real_x=x/self.pixels_per_meter,
                                      real_y=y/self.pixels_per_meter)
                    self.node_positions[node_name] = (x, y)
                    nodes.append((node_name, x, y))
                    node_id += 1
        
        print(f"✅ Created {len(nodes)} navigation nodes")
        return nodes
    
    def connect_nodes(self, max_connection_distance: float = None):
        """
        Step 3: Connect adjacent nodes with weighted edges
        
        Args:
            max_connection_distance: Maximum distance to connect nodes (pixels)
        """
        if max_connection_distance is None:
            max_connection_distance = 30  # Default connection distance
        
        nodes = list(self.graph.nodes())
        connections = 0
        
        for i, node1 in enumerate(nodes):
            pos1 = self.node_positions[node1]
            
            for j, node2 in enumerate(nodes[i+1:], i+1):
                pos2 = self.node_positions[node2]
                distance = euclidean(pos1, pos2)
                
                if distance <= max_connection_distance:
                    # Check if path between nodes is clear
                    if self._is_path_clear(pos1, pos2):
                        # Convert pixel distance to real-world distance
                        real_distance = distance / self.pixels_per_meter
                        
                        self.graph.add_edge(node1, node2, 
                                          weight=real_distance,
                                          pixel_distance=distance)
                        connections += 1
        
        print(f"✅ Created {connections} node connections")
        return connections
    
    def _is_path_clear(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if straight line path between two points is walkable"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Use Bresenham's line algorithm to check all pixels along path
        points = self._bresenham_line(x1, y1, x2, y2)
        
        for x, y in points:
            if (x < 0 or y < 0 or 
                y >= self.walkable_mask.shape[0] or 
                x >= self.walkable_mask.shape[1] or
                self.walkable_mask[y, x] == 0):
                return False
        return True
    
    def _bresenham_line(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for pixel-perfect line drawing"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def add_special_locations(self, locations: Dict[str, Tuple[int, int]]):
        """
        Add special locations (entrances, exits, safe rooms, etc.)
        
        Args:
            locations: Dict mapping location names to (x, y) pixel coordinates
        """
        for name, (x, y) in locations.items():
            # Find nearest existing node or create new one
            nearest_node = self._find_nearest_node(x, y)
            
            if nearest_node:
                # Update existing node with location info
                self.graph.nodes[nearest_node]['location_type'] = name
                self.graph.nodes[nearest_node]['is_special'] = True
            else:
                # Create new special node
                node_name = f"LOC_{name.upper().replace(' ', '_')}"
                self.graph.add_node(node_name,
                                  pos_x=x, pos_y=y,
                                  real_x=x/self.pixels_per_meter,
                                  real_y=y/self.pixels_per_meter,
                                  location_type=name,
                                  is_special=True)
                self.node_positions[node_name] = (x, y)
                
                # Connect to nearby nodes
                self._connect_special_node(node_name)
        
        print(f"✅ Added {len(locations)} special locations")
    
    def _find_nearest_node(self, x: int, y: int, max_distance: float = 25) -> Optional[str]:
        """Find the nearest existing node to given coordinates"""
        min_dist = float('inf')
        nearest = None
        
        for node, (nx, ny) in self.node_positions.items():
            dist = euclidean((x, y), (nx, ny))
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _connect_special_node(self, node_name: str, max_connections: int = 5):
        """Connect a special node to nearby regular nodes"""
        pos = self.node_positions[node_name]
        distances = []
        
        for other_node, other_pos in self.node_positions.items():
            if other_node != node_name:
                dist = euclidean(pos, other_pos)
                distances.append((dist, other_node))
        
        # Connect to closest nodes
        distances.sort()
        connections = 0
        
        for dist, other_node in distances[:max_connections]:
            if dist <= 50 and self._is_path_clear(pos, self.node_positions[other_node]):
                real_distance = dist / self.pixels_per_meter
                self.graph.add_edge(node_name, other_node,
                                  weight=real_distance,
                                  pixel_distance=dist)
                connections += 1
        
        return connections
    
    def save_graph(self, output_path: str):
        """Save the generated graph to GEXF format"""
        nx.write_gexf(self.graph, output_path)
        print(f"✅ Graph saved to {output_path}")
    
    def visualize_graph(self, save_path: str = None, show_connections: bool = True):
        """Create a visualization of the generated graph"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original floor plan
        ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Floor Plan")
        ax1.axis('off')
        
        # Processed floor plan with graph
        ax2.imshow(self.walkable_mask, cmap='gray')
        
        # Draw nodes
        for node, (x, y) in self.node_positions.items():
            node_data = self.graph.nodes[node]
            if node_data.get('is_special', False):
                ax2.plot(x, y, 'ro', markersize=8, label='Special Location')
            else:
                ax2.plot(x, y, 'bo', markersize=3, alpha=0.7)
        
        # Draw connections
        if show_connections:
            for edge in self.graph.edges():
                node1, node2 = edge
                pos1 = self.node_positions[node1]
                pos2 = self.node_positions[node2]
                ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                        'g-', alpha=0.3, linewidth=0.5)
        
        ax2.set_title(f"Generated Navigation Graph ({len(self.graph.nodes)} nodes)")
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        plt.show()
        return fig

class EnhancedPathfinder:
    """Enhanced pathfinding with multiple algorithms and optimizations"""
    
    def __init__(self, graph: nx.Graph, node_positions: Dict[str, Tuple[int, int]]):
        self.graph = graph
        self.node_positions = node_positions
    
    def find_optimal_path(self, start: str, end: str, algorithm: str = 'dijkstra') -> Dict:
        """
        Find optimal path using specified algorithm
        
        Args:
            start: Start node name
            end: End node name  
            algorithm: 'dijkstra', 'astar', or 'bidirectional'
        """
        try:
            if algorithm == 'dijkstra':
                path = nx.shortest_path(self.graph, start, end, weight='weight')
                length = nx.shortest_path_length(self.graph, start, end, weight='weight')
            
            elif algorithm == 'astar':
                def heuristic(u, v):
                    pos_u = self.node_positions[u]
                    pos_v = self.node_positions[v]
                    return euclidean(pos_u, pos_v)
                
                path = nx.astar_path(self.graph, start, end, heuristic=heuristic, weight='weight')
                length = nx.astar_path_length(self.graph, start, end, heuristic=heuristic, weight='weight')
            
            elif algorithm == 'bidirectional':
                path = nx.bidirectional_shortest_path(self.graph, start, end, weight='weight')
                length = nx.shortest_path_length(self.graph, start, end, weight='weight')
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Generate turn-by-turn directions
            directions = self._generate_directions(path)
            
            return {
                'status': 'success',
                'path': path,
                'length_meters': length,
                'directions': directions,
                'algorithm': algorithm,
                'node_count': len(path)
            }
            
        except nx.NetworkXNoPath:
            return {
                'status': 'no_path',
                'error': f'No path exists between {start} and {end}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_directions(self, path: List[str]) -> List[str]:
        """Generate human-readable turn-by-turn directions"""
        if len(path) < 2:
            return ["You are already at your destination"]
        
        directions = [f"Start at {path[0]}"]
        
        for i in range(1, len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Calculate direction
            current_pos = self.node_positions[current]
            next_pos = self.node_positions[next_node]
            
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            
            # Determine cardinal direction
            if abs(dx) > abs(dy):
                direction = "east" if dx > 0 else "west"
            else:
                direction = "south" if dy > 0 else "north"
            
            directions.append(f"Head {direction} to {next_node}")
        
        directions.append(f"Arrive at {path[-1]}")
        return directions

def main():
    """Example usage of the enhanced pathfinding system"""
    
    # Initialize processor
    floor_plan_path = "BMHS_FloorPlan.JPG"
    processor = FloorPlanProcessor(floor_plan_path, pixels_per_meter=10.0)
    
    # Step 1: Preprocess floor plan
    print("🔄 Step 1: Preprocessing floor plan...")
    processor.preprocess_floor_plan(threshold_method='adaptive')
    
    # Step 2: Create node grid
    print("🔄 Step 2: Creating navigation nodes...")
    processor.create_node_grid(grid_spacing=25, min_clearance=8)
    
    # Step 3: Connect nodes
    print("🔄 Step 3: Connecting nodes...")
    processor.connect_nodes(max_connection_distance=35)
    
    # Step 4: Add special locations
    print("🔄 Step 4: Adding special locations...")
    special_locations = {
        "Main Entrance": (100, 200),
        "Safe Room 1": (300, 150),
        "Safe Room 2": (500, 300),
        "Cafeteria": (200, 400),
        "Library": (450, 200)
    }
    processor.add_special_locations(special_locations)
    
    # Step 5: Save and visualize
    print("🔄 Step 5: Saving results...")
    processor.save_graph("enhanced_floor_plan.gexf")
    processor.visualize_graph("floor_plan_visualization.png")
    
    # Test pathfinding
    pathfinder = EnhancedPathfinder(processor.graph, processor.node_positions)
    
    # Find path between special locations
    start_node = "LOC_MAIN_ENTRANCE"
    end_node = "LOC_SAFE_ROOM_1"
    
    result = pathfinder.find_optimal_path(start_node, end_node, algorithm='astar')
    
    if result['status'] == 'success':
        print(f"\n🎯 Path found from {start_node} to {end_node}:")
        print(f"   Distance: {result['length_meters']:.1f} meters")
        print(f"   Nodes: {result['node_count']}")
        print("   Directions:")
        for i, direction in enumerate(result['directions'], 1):
            print(f"   {i}. {direction}")
    else:
        print(f"❌ {result['error']}")

if __name__ == "__main__":
    main()
