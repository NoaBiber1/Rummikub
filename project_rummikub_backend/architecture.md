# System Architecture & Development Guidelines: Rummikub Smart Assistant

## Context for AI Agent
You are an Expert Full-Stack Developer and Algorithm Engineer assisting a team of 3 Computer Science students in building their final capstone project. 
The deadline is extremely tight (~1 month). You must prioritize speed of delivery, clean architecture, Separation of Concerns, robust type-checking, and avoiding over-engineering. Do NOT suggest complex DevOps pipelines or intricate UI animations for the MVP.

## 1. Project Overview & Constraints
*   **Goal:** Build a mobile app that takes pictures of a Rummikub board and rack, recognizes the tiles, allows manual user correction, and calculates the optimal move to drop the maximum number of tiles from the rack to the board.
*   **Time Limit:** The Python solver (OR-Tools) MUST have a strict 5-second execution limit.
*   **UI Simplicity:** Do NOT implement complex Drag & Drop mechanics. Use a simple "Tap to Select -> Tap to Move" mechanism for grid interactions.

## 2. Tech Stack (Strictly Adhere)
*   **Frontend (Mobile):** Flutter (Dart) for iOS & Android. Use simple state management (Provider or Riverpod).
*   **Authentication:** Firebase Auth (Use `firebase_ui_auth` for zero-boilerplate UI).
*   **Backend & DB:** Firebase Firestore (NoSQL).
*   **Computer Vision:** YOLOv8 accessed directly via Roboflow Hosted Inference API (called from Flutter). Do NOT build a custom Python server for vision.
*   **Optimization Solver:** Google OR-Tools (CP-SAT Solver) wrapped in a Python Cloud Function or lightweight FastAPI deployed on Google Cloud Run.

## 3. Core Data Models (JSON Definitions)
All communication between Frontend, DB, and the Solver MUST use these exact structures.

### Tile Object
```json
{
  "id": "string (UUID)",
  "color": "string ('RED', 'BLUE', 'BLACK', 'YELLOW', 'JOKER')",
  "value": "int (1-13, or 0 for JOKER)"
}

Set Object (A valid run or group)
JSON
{
  "id": "string (UUID)",
  "type": "string ('RUN', 'GROUP', 'INVALID')",
  "tiles": ["Array of Tile Objects"]
}
Game State Object
JSON
{
  "board": ["Array of Set Objects"],
  "rack": ["Array of Tile Objects"]
}
4. API Contracts
A. Vision API (Roboflow Inference)
Caller: Flutter Frontend.

Input: Base64 Image or direct file upload to Roboflow URL.

Output: Bounding boxes and class labels (Color_Number).

Frontend Logic: Flutter maps the bounding boxes to the Game State Object based on spatial coordinates (y-axis for rows, x-axis for ordering).

B. Solver API (Python Endpoint)
Input: Validated Game State Object JSON.

Output:

JSON
{
  "original_state": "Game State Object",
  "optimal_state": {
    "board": ["Array of Set Objects (New state)"],
    "tiles_used_from_rack": "int",
    "remaining_rack": ["Array of Tile Objects"]
  },
  "is_valid": true,
  "execution_time_ms": 1250
}
Constraint: The CP-SAT solver must maximize tiles_used_from_rack. It must enforce standard Rummikub rules (min 3 tiles per set, valid runs/groups).

5. Firestore Database Schema
Use these collections strictly:

users: uid (PK), display_name, total_tiles_dropped.

games_history: game_id (PK), user_id, timestamp, tiles_dropped, before_state (JSON), after_state (JSON). Used to view past solutions.

global_arena: challenge_id (PK), initial_state (JSON), algorithm_score (int). Used for community challenges.

vision_corrections: correction_id (PK), vision_output (JSON), user_corrected_output (JSON). Used for future model training.

6. App Flow & State Machine (Flutter)
Auth State: Managed by firebase_ui_auth.

Capture State: Camera capture -> Send to Roboflow -> Parse JSON.

Correction / Human-in-the-Loop State:

Render Game State in a GridView.

Interaction: Tap to select a tile -> Tap empty cell to move it. Tap tile -> Open dialog to change color/value.

Local Dart Validation: Run a local validation algorithm to visually highlight invalid sets (e.g., red border) before allowing submission.

Solving State: User taps "Solve" -> Call Solver API -> Display loading indicator.

Result State: Display the optimal_state. Highlight tiles that were moved from the rack. Save to games_history.

Community Challenge State: Fetch a puzzle from global_arena. User uses the same "Correction" UI to arrange tiles manually. Upon clicking "Submit", run local Dart validation and compare score against algorithm_score.

7. Developer Instructions for AI Output
When asked to generate Flutter code, provide complete, copy-pasteable widgets prioritizing functionality over complex styling.

When asked to generate Python OR-Tools code, comment the constraint logic heavily and explicitly include the 5-second timeout parameter (solver.parameters.max_time_in_seconds = 5.0).

Always implement local Dart logic to validate game sets to reduce unnecessary API calls to the solver.