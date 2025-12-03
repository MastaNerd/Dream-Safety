import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO
import threading
import queue
import mediapipe as mp
import json
import asyncio
import websockets
from datetime import datetime
import base64
from typing import Dict, List, Tuple, Optional
import networkx as nx

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity

# Base paths for shared Pathfinding assets
PATHFINDING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Pathfinding")
)
DEFAULT_FLOOR_PLAN = os.path.join(PATHFINDING_DIR, "BMHS_FloorPlan.JPG")
DEFAULT_GEXF = os.path.join(PATHFINDING_DIR, "BMHS_FloorPlan_combined.gexf")

# Camera names to node IDs in the GEXF; adjust if camera placement changes.
CAMERA_TO_NODE: Dict[str, str] = {
    # Lower-level Dining Commons area (floor 1 footprint)
    "Cafeteria": "Floor1_Node_15",
    # Level-one central hallway spine
    "Hallway A": "Floor2_Node_5",
    # Lower-level gymnasium courts
    "Gym": "Floor1_Node_6",
    # Level-two media center/library side
    "Library": "Floor3_Node_10",
}

# --- Simple Map Display ---
class SimpleMapService:
    def __init__(self, floor_plan_image):
        """Initialize simple map service"""
        self.floor_plan = cv2.imread(floor_plan_image)
        if self.floor_plan is None:
            raise FileNotFoundError(f"Floor plan image not found at {floor_plan_image}")
    
    def get_base_floor_plan(self) -> str:
        """Get the base floor plan image"""
        _, buffer = cv2.imencode('.jpg', self.floor_plan, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64


class PathfindingService:
    def __init__(self, gexf_file: str, floorplan_path: str):
        self.G = nx.read_gexf(gexf_file)
        self.floor_plan = cv2.imread(floorplan_path)
        if self.floor_plan is None:
            raise FileNotFoundError(f"Could not read blueprint image: {floorplan_path}")

        for _, _, data in self.G.edges(data=True):
            data['weight'] = float(data.get('weight', 1.0))
            if 'path' not in data:
                data['path'] = '[]'

        self.pos = {}
        self.floor_info = {}
        self.labels = {}

        nodes_data = list(self.G.nodes(data=True))
        nodes_data.sort(key=lambda nd: nd[0])
        for idx_node, (node, data) in enumerate(nodes_data):
            self.pos[node] = (float(data['pos_x']), float(data['pos_y']))
            self.floor_info[node] = data.get('floor', 'Unknown')
            self.labels[node] = data.get('label', node)

    def pretty_node_name(self, node: str) -> str:
        base_label = self.labels.get(node, node)
        prefix = base_label.split('_')[0].upper()
        LOCATION_CONTEXT = {
            'ENTRY': 'Entry', 'CLASS': 'Classroom', 'TP': 'Teacher Planning', 'SPED': 'Special Education',
            'HUB': 'Interdisciplinary Hub', 'LAB': 'Teaching Lab', 'FIT': 'Fitness', 'TH': 'Theater',
            'TRK': 'Track', 'MC': 'Media Center', 'ART': 'Art', 'MUSIC': 'Music', 'BBT': 'Black-Box Theater',
            'MK': 'Makerspace', 'TSHOP': 'Theater Shop', 'GYM': 'Gymnasium', 'LOCK': 'Locker Rooms',
            'KIT': 'Kitchen/Servery', 'DINING': 'Dining Commons', 'PREK': 'Pre-K'
        }
        if prefix in LOCATION_CONTEXT:
            return f"{LOCATION_CONTEXT[prefix]} ({base_label})"
        return base_label

    def find_path(self, start_node: str, end_node: str):
        if start_node == end_node:
            return {
                'status': 'same_node',
                'path': [start_node],
                'distance': 0.0,
                'instructions': ['Already at destination'],
            }
        try:
            path_nodes = nx.shortest_path(self.G, source=start_node, target=end_node, weight='weight')
            path_length = nx.shortest_path_length(self.G, source=start_node, target=end_node, weight='weight')
        except nx.NetworkXNoPath:
            return {'status': 'no_path', 'error': f'No path between {start_node} and {end_node}'}

        instructions = []
        current_floor = self.floor_info.get(path_nodes[0], 'Unknown')
        instructions.append(f"Start at {self.pretty_node_name(path_nodes[0])} on {current_floor}")
        for i in range(1, len(path_nodes)):
            prev_node = path_nodes[i-1]
            node = path_nodes[i]
            new_floor = self.floor_info.get(node, current_floor)
            edge_data = self.G.get_edge_data(prev_node, node, default={})
            is_stair = edge_data.get('is_stair', 'false') == 'true'
            prev_name = self.pretty_node_name(prev_node)
            curr_name = self.pretty_node_name(node)
            if is_stair and new_floor != current_floor:
                text = f"Take stairs from {current_floor} at {prev_name} to {new_floor} at {curr_name}"
                current_floor = new_floor
            else:
                text = f"Go from {prev_name} → {curr_name} on {current_floor}"
            instructions.append(text)
        instructions.append(f"Arrived at {self.pretty_node_name(end_node)} on {current_floor}")
        return {
            'status': 'success',
            'path': path_nodes,
            'distance': path_length,
            'instructions': instructions,
        }

    def render_path_base64(self, path_nodes: list[str]) -> str:
        img = self.floor_plan.copy()
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if not self.G.has_edge(u, v):
                continue
            edge_data = self.G.get_edge_data(u, v)
            color = (255, 0, 255) if edge_data.get('is_stair', 'false') == 'true' else (0, 0, 255)
            try:
                pts_list = json.loads(edge_data.get('path', '[]'))
                pts = np.array([(int(x), int(y)) for x, y in pts_list], dtype=np.int32)
                cv2.polylines(img, [pts], False, color, 3)
            except Exception:
                p1 = (int(self.pos[u][0]), int(self.pos[u][1]))
                p2 = (int(self.pos[v][0]), int(self.pos[v][1]))
                cv2.line(img, p1, p2, color, 3)
        if path_nodes:
            start = (int(self.pos[path_nodes[0]][0]), int(self.pos[path_nodes[0]][1]))
            end = (int(self.pos[path_nodes[-1]][0]), int(self.pos[path_nodes[-1]][1]))
            cv2.circle(img, start, 9, (0, 255, 0), -1)
            cv2.circle(img, end, 9, (0, 0, 255), -1)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode('utf-8')

    def render_node_highlight_base64(self, node_id: str) -> Optional[str]:
        """Render a single node highlight for quick map updates."""
        if node_id not in self.pos:
            return None
        img = self.floor_plan.copy()
        x, y = int(self.pos[node_id][0]), int(self.pos[node_id][1])
        cv2.circle(img, (x, y), 18, (0, 0, 0), 6, lineType=cv2.LINE_AA)  # halo
        cv2.circle(img, (x, y), 12, (0, 0, 255), -1, lineType=cv2.LINE_AA)  # red fill
        cv2.circle(img, (x, y), 12, (255, 255, 255), 2, lineType=cv2.LINE_AA)  # outline
        label = self.pretty_node_name(node_id)
        cv2.putText(img, label, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(img, label, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode('utf-8')


# --- Detection Helper Functions ---
def find_pose_for_face(face_bbox, pose_landmarks, frame_shape):
    """Finds the pose that corresponds to a given face bounding box."""
    if not pose_landmarks:
        return None
    
    face_x1, face_y1, face_x2, face_y2 = face_bbox
    nose = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    
    nose_px = int(nose.x * frame_shape[1]), int(nose.y * frame_shape[0])

    if face_x1 <= nose_px[0] <= face_x2 and face_y1 <= nose_px[1] <= face_y2:
        return pose_landmarks
    return None

def is_gun_near_hand(gun_bbox, hand_landmark, frame_shape, proximity_thresh=100):
    """Checks if a gun's center is near a hand landmark."""
    if not hand_landmark or hand_landmark.visibility < 0.5:
        return False

    gun_center_x = (gun_bbox[0] + gun_bbox[2]) / 2
    gun_center_y = (gun_bbox[1] + gun_bbox[3]) / 2

    hand_px = int(hand_landmark.x * frame_shape[1]), int(hand_landmark.y * frame_shape[0])

    distance = np.sqrt((gun_center_x - hand_px[0])**2 + (gun_center_y - hand_px[1])**2)
    return distance < proximity_thresh

# --- Combined Detector with Simple Map ---
class CombinedDetectorWithMap:
    def __init__(self, det_weight, rec_weight, gun_weight, faces_dir, 
                 gexf_file=None, floor_plan_image=None, similarity_thresh=0.4,
                 camera_to_node: Optional[Dict[str, str]] = None):
        # Initialize detection models
        self.face_detector = SCRFD(det_weight)
        self.face_recognizer = ArcFace(rec_weight)
        self.gun_detector = YOLO(gun_weight)
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        self.similarity_thresh = similarity_thresh
        self.known_face_embs = []
        self.known_face_names = []
        self.identified_threats = set()
        
        # Initialize map services (prefer shared Pathfinding assets one level up)
        default_gexf = gexf_file or DEFAULT_GEXF
        default_plan = floor_plan_image or DEFAULT_FLOOR_PLAN
        self.map_service = SimpleMapService(default_plan)
        self.path_service = PathfindingService(default_gexf, default_plan)
        self.camera_to_node = camera_to_node or CAMERA_TO_NODE
        self.camera_locations = ["Cafeteria", "Hallway A", "Gym", "Library"]
        self.camera_index = 0
        
        # WebSocket clients
        self.websocket_clients = set()
        

        # Load known faces
        self._load_known_faces(faces_dir)

    def _load_known_faces(self, faces_dir):
        if not os.path.isdir(faces_dir):
            print(f"Error: Faces directory not found at {faces_dir}")
            return

        print("Loading known faces...")
        for image_path in glob.glob(os.path.join(faces_dir, "*")):
            try:
                img = cv2.imread(image_path)
                if img is None: continue
                
                bboxes, kpss = self.face_detector.detect(img)
                if bboxes.shape[0] > 0:
                    face_kps = kpss[0]
                    emb = self.face_recognizer(img, face_kps)
                    face_name = os.path.splitext(os.path.basename(image_path))[0]
                    self.known_face_embs.append(emb)
                    self.known_face_names.append(face_name)
                    print(f"- Loaded face for {face_name}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        print(f"Finished loading {len(self.known_face_names)} known faces.")

    def camera_node_id(self, camera_name: str) -> Optional[str]:
        """Return the mapped node id for a camera if present in the graph."""
        node_id = self.camera_to_node.get(camera_name)
        if node_id and self.path_service.G.has_node(node_id):
            return node_id
        return None

    def current_camera(self) -> str:
        if not self.camera_locations:
            return "Unknown"
        return self.camera_locations[self.camera_index % len(self.camera_locations)]

    def switch_camera(self, direction: str = "next", target: Optional[str] = None) -> str:
        """Cycle cameras or jump to a specific camera name."""
        if target and target in self.camera_locations:
            self.camera_index = self.camera_locations.index(target)
        elif direction == "prev":
            self.camera_index = (self.camera_index - 1) % len(self.camera_locations)
        else:
            self.camera_index = (self.camera_index + 1) % len(self.camera_locations)
        return self.current_camera()

    def detect_and_recognize(self, frame, gun_conf_thresh=0.7):
        # --- Face Recognition ---
        face_detections = []
        bboxes, kpss = self.face_detector.detect(frame)
        if bboxes.shape[0] > 0:
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, :4]
                kps = kpss[i]
                embedding = self.face_recognizer(frame, kps)
                
                best_sim = -1
                best_name = "Unknown"
                for idx, known_emb in enumerate(self.known_face_embs):
                    sim = compute_similarity(embedding, known_emb)
                    if sim > best_sim:
                        best_sim = sim
                        if sim > self.similarity_thresh:
                            best_name = self.known_face_names[idx]
                
                label = f"{best_name} ({best_sim:.2f})"
                face_detections.append((bbox, label))

        # --- Gun Detection ---
        gun_results = self.gun_detector(frame, conf=gun_conf_thresh, verbose=False)[0]
        gun_detections = []
        for box in gun_results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            label = f'{self.gun_detector.model.names[int(box.cls[0])]} {box.conf[0]:.2f}'
            gun_detections.append((xyxy, label))

        # --- Pose Estimation ---
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.mp_pose.process(frame_rgb)
        frame.flags.writeable = True
        all_keypoints = pose_results.pose_landmarks

        # --- Threat Association Logic ---
        final_face_detections = []
        for face_bbox, label in face_detections:
            person_name = label.split(' (')[0]

            if person_name != "Unknown":
                person_pose = find_pose_for_face(face_bbox, pose_results.pose_landmarks, frame.shape)
                if person_pose:
                    lwrist = person_pose.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                    rwrist = person_pose.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

                    for gun_bbox, _ in gun_detections:
                        if is_gun_near_hand(gun_bbox, lwrist, frame.shape) or \
                           is_gun_near_hand(gun_bbox, rwrist, frame.shape):
                            self.identified_threats.add(person_name)
                            break 
            
            is_shooter = person_name in self.identified_threats
            final_face_detections.append((face_bbox, label, is_shooter))

        return final_face_detections, gun_detections, all_keypoints

    async def register_client(self, websocket):
        """Register a new WebSocket client."""
        self.websocket_clients.add(websocket)
        print(f"New client connected. Total clients: {len(self.websocket_clients)}")
        
        # Send base floor plan to new client
        try:
            base_floor_plan = self.map_service.get_base_floor_plan()
            await websocket.send(json.dumps({
                "type": "floor_plan_init",
                "floor_plan": base_floor_plan
            }))
        except Exception as e:
            print(f"Error sending base floor plan to new client: {e}")

    async def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        self.websocket_clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.websocket_clients)}")

    async def broadcast_data(self, data):
        """Broadcast data to all connected WebSocket clients."""
        if self.websocket_clients:
            message = json.dumps(data)
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            for client in disconnected_clients:
                await self.unregister_client(client)
    
    async def handle_simple_requests(self, request_data, websocket=None):
        """Handle simple requests from the dashboard"""
        request_type = request_data.get("type") or request_data.get("request_type")
        
        if request_type == "get_floor_plan":
            floor_plan = self.map_service.get_base_floor_plan()
            payload = {"type": "floor_plan_init", "floor_plan": floor_plan}
            if websocket:
                await websocket.send(json.dumps(payload))
            else:
                await self.broadcast_data(payload)
        elif request_type == "pathfinding_request":
            start = request_data.get("start_node") or request_data.get("start") or request_data.get("officer_node")
            end = request_data.get("end_node") or request_data.get("end") or request_data.get("threat_node")
            if not start or not end:
                response = {"type": "pathfinding_response", "status": "error", "error": "start/end required"}
            else:
                result = self.path_service.find_path(start, end)
                if result.get("status") == "success":
                    img_b64 = self.path_service.render_path_base64(result.get("path", []))
                    response = {
                        "type": "pathfinding_response",
                        "status": "success",
                        "path": result.get("path"),
                        "distance": result.get("distance"),
                        "instructions": result.get("instructions"),
                        "floor_plan": img_b64,
                        "start": start,
                        "end": end,
                    }
                else:
                    response = {"type": "pathfinding_response", **result}
            if websocket:
                await websocket.send(json.dumps(response))
            else:
                await self.broadcast_data(response)

def create_detection_payload(face_detections, gun_detections, frame, location="Main Campus"):
    """Create a JSON payload with detection results."""
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    threats = []
    persons = []
    
    for bbox, label, is_shooter in face_detections:
        person_name = label.split(' (')[0]
        confidence = float(label.split('(')[1].split(')')[0])
        
        person_data = {
            "name": person_name,
            "confidence": confidence,
            "bbox": bbox.tolist(),
            "is_threat": is_shooter,
            "location": location
        }
        
        if is_shooter:
            threats.append({
                "type": "armed_person",
                "person": person_name,
                "confidence": confidence,
                "location": location,
                "timestamp": datetime.now().isoformat()
            })
        
        persons.append(person_data)
    
    timestamp = datetime.now().isoformat()
    weapons = []
    for bbox, label in gun_detections:
        confidence = float(label.split()[-1])
        weapons.append({
            "type": "weapon",
            "weapon_type": label.split()[0],
            "confidence": confidence,
            "bbox": bbox.tolist(),
            "location": location,
            "timestamp": timestamp
        })

    # Generate threats based on detections
    if weapons:
        recognized_persons = [p for p in persons if p["name"] != "Unknown"]
        armed_person_threats = [p for p in persons if p.get("is_threat")]
        first_weapon = weapons[0]

        # Prioritize explicitly identified shooters
        if armed_person_threats:
            for person in armed_person_threats:
                # Avoid duplicate threats if already added
                if not any(t["type"] == "armed_person" and t["person"] == person["name"] for t in threats):
                    threats.append({
                        "type": "armed_person",
                        "person": person["name"],
                        "weapon_type": first_weapon["weapon_type"],
                        "confidence": person["confidence"],
                        "location": location,
                        "timestamp": timestamp
                    })
        # If a gun is detected and a known person is in frame, associate it
        elif recognized_persons:
            person = recognized_persons[0]
            threats.append({
                "type": "armed_person",
                "person": person["name"],
                "weapon_type": first_weapon["weapon_type"],
                "confidence": person["confidence"],
                "location": location,
                "timestamp": timestamp
            })
        # Otherwise, create a general weapon threat
        else:
            threats.append({
                "type": "weapon_detected",
                "weapon_type": first_weapon["weapon_type"],
                "confidence": first_weapon["confidence"],
                "location": location,
                "timestamp": timestamp
            })
    
    return {
        "type": "detection_data",
        "timestamp": timestamp,
        "location": location,
        "frame": frame_base64,
        "detections": {
            "persons": persons,
            "weapons": weapons,
            "threat_count": len(threats)
        },
        "threats": threats,
        "status": "alert" if threats else "clear"
    }

async def websocket_handler(websocket, detector):
    """Handle WebSocket connections and messages."""
    await detector.register_client(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Handle different message types
                if data.get("type") == "pathfinding_request":
                    await detector.handle_simple_requests(data, websocket)
                    
                elif data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                elif data.get("type") == "switch_camera":
                    direction = data.get("direction", "next")
                    target = data.get("camera")
                    new_cam = detector.switch_camera(direction=direction, target=target)
                    await websocket.send(json.dumps({"type": "camera_switched", "camera": new_cam}))
                    
            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message}")
            except Exception as e:
                print(f"Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await detector.unregister_client(websocket)

async def process_video_stream(detector, gun_conf_thresh=0.75):
    """Process video stream and send detection results via WebSocket."""
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    print("\nStarting webcam feed with WebSocket server...")
    
    PROC_WIDTH = 640
    PROC_HEIGHT = 480
    
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()
    
    def worker(frame_queue, result_queue, detector, gun_conf_thresh):
        while True:
            try:
                frame = frame_queue.get()
                if frame is None:
                    break
                face_detections, gun_detections, all_keypoints = detector.detect_and_recognize(
                    frame, gun_conf_thresh=gun_conf_thresh
                )
                result_queue.put((face_detections, gun_detections, all_keypoints))
                frame_queue.task_done()
            except queue.Empty:
                continue
    
    thread = threading.Thread(
        target=worker, 
        args=(frame_queue, result_queue, detector, gun_conf_thresh), 
        daemon=True
    )
    thread.start()
    
    face_detections, gun_detections, all_keypoints = [], [], None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 0)
            
            orig_h, orig_w = frame.shape[:2]
            scale_w, scale_h = orig_w / PROC_WIDTH, orig_h / PROC_HEIGHT
            
            if frame_queue.empty():
                resized_frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))
                frame_queue.put(resized_frame)
            
            try:
                face_detections, gun_detections, all_keypoints = result_queue.get_nowait()
                
                # Get current camera location
                current_location = detector.current_camera()
                
                # Send detection data
                detection_data = create_detection_payload(
                    face_detections, 
                    gun_detections, 
                    resized_frame,
                    location=current_location
                )
                
                await detector.broadcast_data(detection_data)

                # Highlight the camera's node when a threat is present
                if detection_data.get("threats"):
                    node_id = detector.camera_node_id(current_location)
                    if node_id:
                        highlight_img = detector.path_service.render_node_highlight_base64(node_id)
                        await detector.broadcast_data({
                            "type": "camera_threat_highlight",
                            "camera": current_location,
                            "node_id": node_id,
                            "floor": detector.path_service.floor_info.get(node_id, "Unknown"),
                            "label": detector.path_service.pretty_node_name(node_id),
                            "floor_plan": highlight_img  # optional: dashboard can ignore if it prefers client-side rendering
                        })
                
            except queue.Empty:
                pass
            
            # Draw on frame for local display
            if all_keypoints:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, all_keypoints, mp.solutions.pose.POSE_CONNECTIONS
                )
            
            if face_detections:
                for bbox, label, is_shooter in face_detections:
                    x1, y1, x2, y2 = [int(i) for i in bbox]
                    box_color = (0, 0, 255) if is_shooter else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1*scale_w), int(y1*scale_h)), 
                                (int(x2*scale_w), int(y2*scale_h)), box_color, 2)
                    cv2.putText(frame, label, (int(x1*scale_w), int(y1*scale_h) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            if gun_detections:
                for xyxy, label in gun_detections:
                    x1, y1, x2, y2 = [int(i) for i in xyxy]
                    cv2.rectangle(frame, (int(x1*scale_w), int(y1*scale_h)), 
                                (int(x2*scale_w), int(y2*scale_h)), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(x1*scale_w), int(y1*scale_h) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Combined Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Switch camera location (for testing)
                detector.switch_camera("next")
                print(f"Switched to camera: {detector.current_camera()}")
            
            await asyncio.sleep(0.1)
            
    finally:
        frame_queue.put(None)
        thread.join()
        cap.release()
        cv2.destroyAllWindows()

async def main():
    # --- Configuration ---
    DET_WEIGHT = "./weights/det_10g.onnx"
    REC_WEIGHT = "./weights/w600k_r50.onnx"
    GUN_WEIGHT = "./best.pt"
    FACES_DIR = "./faces"
    
    # Floor plan configuration
    FLOOR_PLAN_IMAGE = DEFAULT_FLOOR_PLAN
    GEXF_FILE = DEFAULT_GEXF
    
    GUN_CONF_THRESHOLD = 0.83
    WEBSOCKET_PORT = 8766
    
    detector = CombinedDetectorWithMap(
        det_weight=DET_WEIGHT,
        rec_weight=REC_WEIGHT,
        gun_weight=GUN_WEIGHT,
        faces_dir=FACES_DIR,
        gexf_file=GEXF_FILE,
        floor_plan_image=FLOOR_PLAN_IMAGE,
        camera_to_node=CAMERA_TO_NODE
    )
    
    # Start WebSocket server
    server = await websockets.serve(
        lambda ws: websocket_handler(ws, detector),
        "localhost",
        WEBSOCKET_PORT
    )
    
    print(f"WebSocket server started on ws://localhost:{WEBSOCKET_PORT}")
    print("Press 'c' to cycle through camera locations")
    print("Press 'q' to quit")
    
    # Process video stream
    await process_video_stream(detector, GUN_CONF_THRESHOLD)
    
    # Close server
    server.close()
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
