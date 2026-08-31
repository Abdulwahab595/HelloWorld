"""Canonical schema of the REP-AI temporal dataset.

Every JSON session produced by the REP-AI Android app has the shape::

    {
      "session_id": "session_1778674087576",
      "user_id":    "user_01",
      "exercise":   "bicep_curl",
      "label":      "correct",
      "fps":        30,
      "reps": [ {"rep_id": 1, "frame_count": 92,
                 "frames": [{"frame_id", "timestamp", "features": {...}}],
                 "rep_summary": {...}} ]
    }

This module is the single source of truth for feature order.  The Kotlin
``FeatureExtractor`` on the phone must emit tensors in exactly this order or
on-device inference will silently produce garbage.
"""

# --- frame-level features -------------------------------------------------
# Order is contractual: index i here == channel i of the model input tensor.
FRAME_FEATURES = [
    "knee_angle",
    "hip_angle",
    "elbow_angle",           # left elbow
    "right_elbow_angle",
    "back_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "velocity",
    "avg_elbow_angle",
    "left_right_asymmetry",
    "angular_velocity",
    "angular_acceleration",
]

# `phase` is also present in the JSON but is excluded on purpose: in the
# collected data it is 99.7 % the constant string "up", so it carries no
# information and would only teach the model to ignore a channel.
DEGENERATE_FEATURES = ["phase"]

# Left/right pairs used by the mirror augmentation.  Swapping these simulates
# a user who trains with the opposite arm / faces the camera the other way.
MIRROR_PAIRS = [
    ("elbow_angle", "right_elbow_angle"),
    ("left_shoulder_angle", "right_shoulder_angle"),
]

# --- repetition-level features -------------------------------------------
REP_SUMMARY_FEATURES = [
    "max_elbow_angle",
    "min_elbow_angle",
    "rom_degrees",
    "max_left_right_asymmetry",
    "avg_back_angle",
    "max_back_deviation",
    "curl_rom_degrees",
    "rep_duration_ms",
    "motion_smoothness",
]

# --- label taxonomy -------------------------------------------------------
# Directory name -> canonical label.  The app writes a `label` field too, but
# it disagrees with the folder for one class (folder `incomplete_extension`
# is written as `partial_range` inside the JSON), so the folder wins.
LABELS = {
    "bicep_curl": [
        "correct",
        "elbow_moving",
        "fast_swing",
        "incomplete_extension",
    ],
    "shoulder_press": [
        "correct",
        "uneven_press",
        "back_arch",
        "incomplete_extension",
    ],
    "squat": [
        "correct",
        "shallow_squats",
        "knee_forward",
    ],
}

# Folder name -> exercise key.
EXERCISE_DIRS = {
    "BICEPT_CURL": "bicep_curl",
    "SHOULDER_PRESS": "shoulder_press",
    "SQUATS": "squat",
}

# Human-readable coaching cue per error class, consumed by the feedback
# engine (module 4).  Kept here so the model and the voice prompts can never
# drift out of sync.
COACHING_CUES = {
    "correct": "Good form, keep going.",
    "elbow_moving": "Keep your elbows pinned to your sides.",
    "fast_swing": "Slow down, stop swinging the weight.",
    "incomplete_extension": "Extend all the way at the bottom.",
    "uneven_press": "Press both arms evenly.",
    "back_arch": "Brace your core, stop arching your back.",
    "shallow_squats": "Go deeper, hips below parallel.",
    "knee_forward": "Sit back, keep your knees behind your toes.",
}
