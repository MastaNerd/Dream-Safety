# mapBuilder.py
import cv2
import numpy as np
import networkx as nx
import json
import os
from typing import Dict, Tuple, List, Optional

class MultiFloorEditor:
    """
    Map builder for Dream Safety.

    Controls:
      - 'n' : Add nodes mode (click to drop nodes on CURRENT FLOOR).
      - 'e' : Add edges mode (click-drag between two existing nodes).
      - 's' : Add stair connections (click node on floor A then node on floor B).
      - '1'..'9' : Switch floors by index (Floor1, Floor2, Floor3, ...).
      - 'c' : Save combined graph to <image_basename>_combined.gexf.
      - 'q' : Quit.

    Nodes are saved with attributes:
      - pos_x, pos_y : pixel coordinates on the blueprint
      - floor       : floor name (e.g. "Floor1")
      - label       : human readable label (defaults to node id)

    Edges:
      - weight      : total drawn path length (float)
      - path        : JSON list of (x,y) points along the polyline
      - is_stair    : "true" / "false"
    """

    def __init__(self, image_path: str, floor_names: List[str]):
        self.image_path = image_path
        self.full_image = cv2.imread(image_path)

        if self.full_image is None:
            raise FileNotFoundError(f"Could not read blueprint image: {image_path}")

        self.display_image = self.full_image.copy()

        # floor_name -> {graph, nodes, color}
        self.floors: Dict[str, Dict] = {}
        for name in floor_names:
            self.floors[name] = {
                "graph": nx.Graph(),
                "nodes": {},        # node_id -> (x,y)
                "color": (200, 200, 200),  # keep text readable; drawing colors are fixed below
            }

        self.current_floor: str = floor_names[0]
        self.combined_graph = nx.Graph()
        self.stair_connections: List[Dict] = []

        # drawing state
        self.drawing: bool = False
        self.current_path: List[Tuple[int, int]] = []
        self.stair_selection: Optional[Tuple[str, str]] = None

        # attempt to load existing combined graph if present
        base, _ = os.path.splitext(self.image_path)
        combined_path = f"{base}_combined.gexf"
        if os.path.exists(combined_path):
            print(f"Found existing combined graph: {combined_path}, loading...")
            self.load_combined_graph(combined_path)
        else:
            print("No existing combined graph found, starting fresh.")

    # -------------------------------------------------------------------------
    # Loading / saving
    # -------------------------------------------------------------------------
    def load_combined_graph(self, gexf_path: str) -> bool:
        if not os.path.exists(gexf_path):
            return False

        G = nx.read_gexf(gexf_path)

        # clear
        for floor_data in self.floors.values():
            floor_data["graph"].clear()
            floor_data["nodes"].clear()
        self.stair_connections = []
        self.combined_graph = nx.Graph()

        # nodes
        for node, data in G.nodes(data=True):
            floor_name = data.get("floor")
            if floor_name not in self.floors:
                continue
            x = float(data["pos_x"])
            y = float(data["pos_y"])
            self.floors[floor_name]["nodes"][node] = (x, y)
            self.floors[floor_name]["graph"].add_node(
                node,
                pos_x=x,
                pos_y=y,
                floor=floor_name,
                label=data.get("label", node),
            )

        # edges
        for u, v, data in G.edges(data=True):
            is_stair = data.get("is_stair", "false")
            weight = float(data.get("weight", 1.0))
            path_str = data.get("path", "[]")
            if is_stair == "true":
                # figure out which floors nodes belong to
                floor1 = floor2 = None
                for fname, fdata in self.floors.items():
                    if u in fdata["nodes"]:
                        floor1 = fname
                    if v in fdata["nodes"]:
                        floor2 = fname
                if floor1 and floor2 and floor1 != floor2:
                    self.stair_connections.append(
                        {
                            "floor1": floor1,
                            "node1": u,
                            "floor2": floor2,
                            "node2": v,
                            "weight": weight,
                            "path": path_str,
                        }
                    )
            else:
                for fname, fdata in self.floors.items():
                    if u in fdata["nodes"] and v in fdata["nodes"]:
                        fdata["graph"].add_edge(
                            u,
                            v,
                            weight=weight,
                            path=path_str,
                            is_stair="false",
                        )
                        break

        # rebuild combined graph
        self.combined_graph = nx.compose_all(
            [fdata["graph"] for fdata in self.floors.values()]
        )
        for conn in self.stair_connections:
            self.combined_graph.add_edge(
                conn["node1"],
                conn["node2"],
                weight=float(conn.get("weight", 10.0)),
                path=conn.get("path", "[]"),
                is_stair="true",
            )

        print(f"Loaded {len(self.combined_graph.nodes())} nodes from {gexf_path}")
        return True

    def save_combined_graph(self) -> bool:
        base, _ = os.path.splitext(self.image_path)
        combined_path = f"{base}_combined.gexf"

        G_save = nx.Graph()

        # nodes
        for fname, fdata in self.floors.items():
            for node, (x, y) in fdata["nodes"].items():
                node_data = fdata["graph"].nodes.get(node, {})
                label = node_data.get("label", node)
                G_save.add_node(
                    node,
                    pos_x=float(x),
                    pos_y=float(y),
                    floor=fname,
                    label=label,
                )

        # edges (non-stairs)
        for fname, fdata in self.floors.items():
            for u, v, data in fdata["graph"].edges(data=True):
                G_save.add_edge(
                    u,
                    v,
                    weight=float(data.get("weight", 1.0)),
                    path=data.get("path", "[]"),
                    is_stair="false",
                )

        # stair connections
        for conn in self.stair_connections:
            G_save.add_edge(
                conn["node1"],
                conn["node2"],
                weight=float(conn.get("weight", 10.0)),
                path=conn.get("path", "[]"),
                is_stair="true",
            )

        nx.write_gexf(G_save, combined_path)
        print(f"Saved combined graph to {combined_path}")
        return True

    # -------------------------------------------------------------------------
    # Drawing helpers
    # -------------------------------------------------------------------------
    def get_closest_node(self, x: int, y: int, floor_name: Optional[str] = None) -> Optional[str]:
        floor = self.floors[floor_name or self.current_floor]
        if not floor["nodes"]:
            return None

        closest = None
        min_d = float("inf")
        for node, (nx_, ny_) in floor["nodes"].items():
            d = np.sqrt((nx_ - x) ** 2 + (ny_ - y) ** 2)
            if d < min_d:
                min_d = d
                closest = node

        # snap only if reasonably close
        return closest if min_d < 20 else None

    def add_stair_connection(
        self,
        floor1: str,
        node1: str,
        floor2: str,
        node2: str,
        weight: float = 10.0,
        path: str = "[]",
    ) -> bool:
        if (
            floor1 not in self.floors
            or floor2 not in self.floors
            or node1 not in self.floors[floor1]["nodes"]
            or node2 not in self.floors[floor2]["nodes"]
        ):
            return False

        self.combined_graph.add_edge(
            node1, node2, weight=float(weight), path=path, is_stair="true"
        )
        self.stair_connections.append(
            {
                "floor1": floor1,
                "node1": node1,
                "floor2": floor2,
                "node2": node2,
                "weight": float(weight),
                "path": path,
            }
        )
        return True

    def redraw_display(self):
        self.display_image = self.full_image.copy()

        # draw only current floor to avoid cross-floor clutter
        fdata = self.floors[self.current_floor]
        # edges (always gray)
        for u, v, data in fdata["graph"].edges(data=True):
            path_str = data.get("path", "[]")
            try:
                path = json.loads(path_str)
                pts = np.array([(int(x), int(y)) for x, y in path], dtype=np.int32)
                cv2.polylines(self.display_image, [pts], False, (160, 160, 160), 2)
            except Exception:
                pass

        # nodes (always blue, stairs tinted purple)
        for node, (x, y) in fdata["nodes"].items():
            node_color = (255, 0, 0)  # blue
            for conn in self.stair_connections:
                if conn["node1"] == node or conn["node2"] == node:
                    node_color = (255, 0, 255)  # purple for stairs
                    break
            cv2.circle(self.display_image, (int(x), int(y)), 6, node_color, -1)

        cv2.putText(
            self.display_image,
            f"Current Floor: {self.current_floor}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
        )
        cv2.imshow("Multi-Floor Editor", self.display_image)

    # -------------------------------------------------------------------------
    # Main editor loop
    # -------------------------------------------------------------------------
    def run(self):
        cv2.namedWindow("Multi-Floor Editor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Floor Editor", 1200, 800)

        mode = "add_nodes"
        self.redraw_display()

        print("Controls:")
        print("  n : Add nodes")
        print("  e : Add edges")
        print("  s : Add stairs (between floors)")
        print("  1-9 : Switch floor")
        print("  c : Save combined graph")
        print("  q : Quit")

        def mouse_cb(event, x, y, flags, param):
            nonlocal mode
            if event == cv2.EVENT_LBUTTONDOWN:
                if mode == "add_nodes":
                    node_id = f"{self.current_floor}_Node_{len(self.floors[self.current_floor]['nodes']) + 1}"
                    self.floors[self.current_floor]["nodes"][node_id] = (x, y)
                    self.floors[self.current_floor]["graph"].add_node(
                        node_id,
                        pos_x=float(x),
                        pos_y=float(y),
                        floor=self.current_floor,
                        label=node_id,
                    )
                    self.combined_graph.add_node(
                        node_id,
                        pos_x=float(x),
                        pos_y=float(y),
                        floor=self.current_floor,
                        label=node_id,
                    )
                    self.redraw_display()

                elif mode == "add_edges":
                    self.drawing = True
                    self.current_path = [(x, y)]
                    self.redraw_display()
                    cv2.circle(self.display_image, (x, y), 3, (0, 0, 255), -1)
                    cv2.imshow("Multi-Floor Editor", self.display_image)

                elif mode == "add_stairs":
                    node = self.get_closest_node(x, y)
                    if node:
                        if not self.stair_selection:
                            self.stair_selection = (self.current_floor, node)
                            print(f"Selected {node} on {self.current_floor}")
                        else:
                            f1, n1 = self.stair_selection
                            f2, n2 = self.current_floor, node
                            if f1 == f2 and n1 == n2:
                                print("Cannot connect a node to itself.")
                            else:
                                self.add_stair_connection(f1, n1, f2, n2)
                                print(f"Connected {n1} ({f1}) <-> {n2} ({f2})")
                                self.redraw_display()
                            self.stair_selection = None

            elif event == cv2.EVENT_MOUSEMOVE and self.drawing and mode == "add_edges":
                cv2.line(self.display_image, self.current_path[-1], (x, y), (255, 0, 0), 2)
                self.current_path.append((x, y))
                cv2.imshow("Multi-Floor Editor", self.display_image)

            elif event == cv2.EVENT_LBUTTONUP and self.drawing and mode == "add_edges":
                self.drawing = False
                if len(self.current_path) >= 2:
                    start_node = self.get_closest_node(*self.current_path[0])
                    end_node = self.get_closest_node(*self.current_path[-1])
                    if start_node and end_node:
                        # distance along the polyline
                        dist = 0.0
                        for i in range(1, len(self.current_path)):
                            x1, y1 = self.current_path[i - 1]
                            x2, y2 = self.current_path[i]
                            dist += np.linalg.norm(
                                np.array([x2, y2]) - np.array([x1, y1])
                            )
                        path_str = json.dumps(self.current_path)
                        self.floors[self.current_floor]["graph"].add_edge(
                            start_node,
                            end_node,
                            weight=round(float(dist), 1),
                            path=path_str,
                            is_stair="false",
                        )
                        self.combined_graph.add_edge(
                            start_node,
                            end_node,
                            weight=round(float(dist), 1),
                            path=path_str,
                            is_stair="false",
                        )
                        self.redraw_display()
                self.current_path = []

        cv2.setMouseCallback("Multi-Floor Editor", mouse_cb)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                mode = "add_nodes"
                self.stair_selection = None
                print("Mode: Add Nodes")
            elif key == ord("e"):
                mode = "add_edges"
                self.stair_selection = None
                print("Mode: Add Edges")
            elif key == ord("s"):
                mode = "add_stairs"
                self.stair_selection = None
                print("Mode: Add Stairs")
            elif key == ord("c"):
                self.save_combined_graph()
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                floor_names = list(self.floors.keys())
                if 0 <= idx < len(floor_names):
                    self.current_floor = floor_names[idx]
                    self.redraw_display()
                    print(f"Switched to {self.current_floor}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(script_dir, "BMHS_FloorPlan.JPG")
    FLOOR_NAMES = ["Floor1", "Floor2", "Floor3"]  # adjust if needed

    editor = MultiFloorEditor(IMAGE_PATH, FLOOR_NAMES)
    editor.run()
