# pathfinder.py
import base64
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import cv2
import networkx as nx
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEXF_FILE = os.path.join(SCRIPT_DIR, "BMHS_FloorPlan_combined.gexf")
FLOORPLAN_IMAGE = os.path.join(SCRIPT_DIR, "BMHS_FloorPlan.JPG")

# Map of area codes -> human-readable names
LOCATION_CONTEXT = {
    "ENTRY": "Entry",
    "CLASS": "Classroom",
    "TP": "Teacher Planning",
    "SPED": "Special Education",
    "HUB": "Interdisciplinary Hub",
    "LAB": "Teaching Lab",
    "FIT": "Fitness",
    "TH": "Theater",
    "TRK": "Track",
    "MC": "Media Center",
    "ART": "Art",
    "MUSIC": "Music",
    "BBT": "Black-Box Theater",
    "MK": "Makerspace",
    "TSHOP": "Theater Shop",
    "GYM": "Gymnasium",
    "LOCK": "Locker Rooms",
    "KIT": "Kitchen/Servery",
    "DINING": "Dining Commons",
    "PREK": "Pre-K",
}

FRIENDLY_BASE_NAMES = [
    "Atrium Corner",
    "Garden Hall",
    "Harbor Bend",
    "Library Nook",
    "Workshop Lane",
    "Gallery Walk",
    "Commons Curve",
    "Lantern Point",
    "Courtyard Edge",
    "Bridge Hall",
    "Studio Loft",
    "Beacon Hall",
    "Skyway",
    "Arcade",
    "Lounge Turn",
    "Reading Corner",
    "Study Hub",
    "Compass Hall",
    "Landing",
    "Pavilion",
]

MOVE_TEMPLATES = [
    "Move to {to}",
    "Head toward {to}",
    "Continue to {to}",
    "Walk to {to}",
]

STAIR_UP_TEMPLATES = [
    "Go up the staircase toward {to_floor}",
    "Take the stairs up to {to_floor}",
    "Go up the {from_floor} stairwell to {to_floor}",
    "Climb to {to_floor} via the stairs",
]

STAIR_DOWN_TEMPLATES = [
    "Go down the staircase toward {to_floor}",
    "Take the stairs down to {to_floor}",
    "Descend from {from_floor} to {to_floor}",
    "Head down the stairwell to {to_floor}",
]

class PathfindingService:
    def __init__(self, gexf_file: str, floorplan_path: str):
        self.G = nx.read_gexf(gexf_file)
        self.floor_plan = cv2.imread(floorplan_path)
        if self.floor_plan is None:
            raise FileNotFoundError(f"Could not read blueprint image: {floorplan_path}")

        for _, _, data in self.G.edges(data=True):
            data["weight"] = float(data.get("weight", 1.0))
            if "path" not in data:
                data["path"] = "[]"

        self.pos: Dict[str, Tuple[float, float]] = {}
        self.floor_info: Dict[str, str] = {}
        self.labels: Dict[str, str] = {}

        nodes_data = list(self.G.nodes(data=True))
        nodes_data.sort(key=lambda nd: nd[0])

        for idx, (node, data) in enumerate(nodes_data):
            self.pos[node] = (float(data["pos_x"]), float(data["pos_y"]))
            self.floor_info[node] = data.get("floor", "Unknown")

            raw_label = data.get("label", node)
            if "Node" in raw_label or raw_label.startswith("Floor"):
                base = FRIENDLY_BASE_NAMES[idx % len(FRIENDLY_BASE_NAMES)]
                floor = self.floor_info[node]
                raw_label = f"{base} ({floor})"
            self.labels[node] = raw_label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def pretty_node_name(self, node: str) -> str:
        """Use label attribute, falling back to generated friendly names."""
        base_label = self.labels.get(node, node)

        # Try to extract a semantic prefix if you use naming like 'GYM_F1_01'
        prefix = base_label.split("_")[0].upper()
        if prefix in LOCATION_CONTEXT:
            return f"{LOCATION_CONTEXT[prefix]} ({base_label})"
        return base_label

    def get_nearest_node(self, x: int, y: int, max_dist: float = 40) -> Optional[str]:
        closest = None
        min_d = float("inf")
        for node, (nx_, ny_) in self.pos.items():
            d = np.sqrt((nx_ - x) ** 2 + (ny_ - y) ** 2)
            if d < min_d and d < max_dist:
                min_d = d
                closest = node
        return closest

    def draw_graph_overlay(self, draw_labels: bool = False) -> np.ndarray:
        """Render all nodes/edges on top of the blueprint for visibility."""
        img = self.floor_plan.copy()

        # edges first
        for u, v, data in self.G.edges(data=True):
            color = (160, 160, 160)  # gray for all edges
            if "path" in data:
                try:
                    pts_list = json.loads(data["path"])
                    pts = np.array([(int(x), int(y)) for x, y in pts_list], dtype=np.int32)
                    # soft outline + crisp inner line to reduce jaggedness
                    cv2.polylines(img, [pts], False, (210, 210, 210), 4, lineType=cv2.LINE_AA)
                    cv2.polylines(img, [pts], False, color, 2, lineType=cv2.LINE_AA)
                except Exception:
                    pass
            else:
                p1 = (int(self.pos[u][0]), int(self.pos[u][1]))
                p2 = (int(self.pos[v][0]), int(self.pos[v][1]))
                cv2.line(img, p1, p2, (210, 210, 210), 4, lineType=cv2.LINE_AA)
                cv2.line(img, p1, p2, color, 2, lineType=cv2.LINE_AA)

        # nodes on top
        for node, (x, y) in self.pos.items():
            center = (int(x), int(y))
            cv2.circle(img, center, 8, (240, 240, 240), -1, lineType=cv2.LINE_AA)  # halo
            cv2.circle(img, center, 5, (255, 0, 0), -1, lineType=cv2.LINE_AA)  # blue nodes
            if draw_labels:
                label = self.pretty_node_name(node)
                cv2.putText(
                    img,
                    label,
                    (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )

        return img

    # ------------------------------------------------------------------
    # Core pathfinding
    # ------------------------------------------------------------------
    def find_path_with_steps(self, start_node: str, end_node: str) -> Dict:
        if start_node == end_node:
            return {
                "status": "same_node",
                "path": [start_node],
                "distance": 0.0,
                "instructions": ["Already at destination"],
                "step_segments": [],
            }

        try:
            path_nodes = nx.shortest_path(
                self.G, source=start_node, target=end_node, weight="weight"
            )
            path_length = nx.shortest_path_length(
                self.G, source=start_node, target=end_node, weight="weight"
            )
        except nx.NetworkXNoPath:
            return {
                "status": "no_path",
                "error": f"No path between {start_node} and {end_node}",
            }

        instructions: List[str] = []
        step_segments: List[Dict] = []

        def floor_num(floor: str) -> Optional[int]:
            m = re.search(r"(\d+)", floor)
            return int(m.group(1)) if m else None

        def choose_template(templates: List[str], key: str) -> str:
            return templates[abs(hash(key)) % len(templates)]

        current_floor = self.floor_info.get(path_nodes[0], "Unknown")
        start_name = self.pretty_node_name(path_nodes[0])
        instructions.append(f"Start at {start_name} on {current_floor}")

        for i in range(1, len(path_nodes)):
            prev_node = path_nodes[i - 1]
            node = path_nodes[i]
            new_floor = self.floor_info.get(node, current_floor)
            edge_data = self.G.get_edge_data(prev_node, node, default={})
            is_stair = edge_data.get("is_stair", "false") == "true"

            curr_name = self.pretty_node_name(node)
            prev_name = self.pretty_node_name(prev_node)

            if is_stair and new_floor != current_floor:
                prev_floor_num = floor_num(current_floor)
                new_floor_num = floor_num(new_floor)
                going_up = (
                    prev_floor_num is not None
                    and new_floor_num is not None
                    and new_floor_num > prev_floor_num
                )
                templates = STAIR_UP_TEMPLATES if going_up else STAIR_DOWN_TEMPLATES
                text = choose_template(
                    templates,
                    f"{prev_node}-{node}-{current_floor}-{new_floor}",
                ).format(
                    from_floor=current_floor,
                    to_floor=new_floor,
                    to=curr_name,
                )
                current_floor = new_floor
            else:
                text = choose_template(
                    MOVE_TEMPLATES, f"{prev_node}-{node}-{current_floor}"
                ).format(to=curr_name)

            instructions.append(text)
            step_segments.append(
                {
                    "from": prev_node,
                    "to": node,
                    "from_label": prev_name,
                    "to_label": curr_name,
                    "floor": current_floor,
                    "text": text,
                }
            )

        end_name = self.pretty_node_name(end_node)
        instructions.append(f"Arrived at {end_name} on {current_floor}")

        return {
            "status": "success",
            "path": path_nodes,
            "distance": path_length,
            "instructions": instructions,
            "step_segments": step_segments,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_path(self, path_nodes: List[str]) -> np.ndarray:
        img = self.draw_graph_overlay()

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if not self.G.has_edge(u, v):
                continue
            edge_data = self.G.get_edge_data(u, v)
            color = (0, 0, 255)  # red for shortest-path overlay

            if "path" in edge_data:
                try:
                    pts_list = json.loads(edge_data["path"])
                    pts = np.array([(int(x), int(y)) for x, y in pts_list], dtype=np.int32)
                    cv2.polylines(img, [pts], False, (0, 0, 180), 6, lineType=cv2.LINE_AA)
                    cv2.polylines(img, [pts], False, color, 3, lineType=cv2.LINE_AA)
                except Exception:
                    pass
            else:
                p1 = (int(self.pos[u][0]), int(self.pos[u][1]))
                p2 = (int(self.pos[v][0]), int(self.pos[v][1]))
                cv2.line(img, p1, p2, (0, 0, 180), 6, lineType=cv2.LINE_AA)
                cv2.line(img, p1, p2, color, 3, lineType=cv2.LINE_AA)

        if path_nodes:
            start = (int(self.pos[path_nodes[0]][0]), int(self.pos[path_nodes[0]][1]))
            end = (int(self.pos[path_nodes[-1]][0]), int(self.pos[path_nodes[-1]][1]))
            cv2.circle(img, start, 12, (240, 240, 240), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, start, 9, (0, 200, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, end, 12, (240, 240, 240), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, end, 9, (0, 0, 200), -1, lineType=cv2.LINE_AA)

        return img

    def render_path_base64(self, path_nodes: List[str]) -> str:
        img = self.render_path(path_nodes)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode("utf-8")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    @staticmethod
    def draw_label_callout(img: np.ndarray, text: str, anchor: Tuple[int, int]) -> None:
        """Draw a small label box near a point."""
        pad = 4
        text = text.strip()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        box_w = tw + pad * 2
        box_h = th + pad * 2
        x, y = anchor
        box_pt1 = (x + 10, y - box_h - 10)
        box_pt2 = (x + 10 + box_w, y - 10)
        cv2.rectangle(img, box_pt1, box_pt2, (245, 245, 245), -1)
        cv2.rectangle(img, box_pt1, box_pt2, (120, 120, 120), 1)
        cv2.putText(
            img,
            text,
            (box_pt1[0] + pad, box_pt2[1] - pad - 2),
            font,
            scale,
            (30, 30, 30),
            thickness,
            lineType=cv2.LINE_AA,
        )


# ----------------------------------------------------------------------
# Simple interactive demo: click start and end on the blueprint
# ----------------------------------------------------------------------
def interactive_demo():
    service = PathfindingService(GEXF_FILE, FLOORPLAN_IMAGE)
    base = service.draw_graph_overlay()
    display = base.copy()

    selected: List[str] = []  # list of node ids: [start, end]

    def mouse_cb(event, x, y, flags, param):
        nonlocal selected, display
        if event == cv2.EVENT_LBUTTONDOWN:
            node = service.get_nearest_node(x, y)
            if node:
                selected.append(node)
                label = service.pretty_node_name(node)
                print(f"Selected node: {node} ({label})")
                display = base.copy()
                cx, cy = int(service.pos[node][0]), int(service.pos[node][1])
                cv2.circle(display, (cx, cy), 12, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(display, (cx, cy), 9, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                service.draw_label_callout(display, label, (cx, cy))
                cv2.imshow("Pathfinder", display)

                if len(selected) == 2:
                    start, end = selected
                    print(f"\n=== Computing path from {start} to {end} ===")
                    result = service.find_path_with_steps(start, end)
                    if result["status"] != "success":
                        print(result)
                    else:
                        print(f"Total distance: {result['distance']:.1f} units")
                        print("\nStep-by-step instructions:")
                        for i, step in enumerate(result["step_segments"], start=1):
                            print(f"{i}. {step['text']}")

                        # draw path
                        path_img = service.render_path(result["path"])
                        cv2.imshow("Pathfinder", path_img)
                    # reset for next selection
                    selected = []

    cv2.namedWindow("Pathfinder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pathfinder", 1200, 800)
    cv2.setMouseCallback("Pathfinder", mouse_cb)
    cv2.imshow("Pathfinder", display)

    print("Click a start node, then an end node on the map.")
    print("Press 'q' to quit.\n")

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    interactive_demo()
