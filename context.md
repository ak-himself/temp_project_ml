# Context Log

## Date
- 2026-04-06

## Scope Executed
- Step 1 only (modified)
- No Step 2+ implementation changes in this update

## Step 1 Redesign Summary

Implemented final Step 1 capture contract:

1. Dual output per sample
- RGB ROI crop saved to:
  - data/rgb/<CLASS>/imgXXXXXX.png
- 117D feature vector saved to:
  - data/landmarks/<CLASS>/lmXXXXXX.npy
- Strict 1:1 mapping via shared numeric sample_id.

2. 117D feature schema
- [63 landmarks + 15 joint angles + 30 distances + 5 finger states + 4 palm direction]

3. Subject removal
- Removed subject_id/session usage from capture path and log schema.
- Capture log now records only sample-centric metadata.

4. Skeleton policy
- Skeleton not saved as dataset artifact.
- Skeleton rendering retained for live preview/debug overlay only.

5. Finger-state approach
- Implemented 3-level finger state output:
  - 0 closed
  - 1 half-open
  - 2 open
- Uses placeholder thresholds in config.py for manual future tuning.
- No calibration workflow implemented.

6. Palm direction feature
- Palm normal computed using:
  - v1 = index_mcp - wrist
  - v2 = pinky_mcp - wrist
  - normal = cross(v1, v2)
- 4-class one-hot used in feature vector:
  - camera, face, up, down
- Extended 6-class orientation used for analysis overlay/log:
  - camera, face, up, down, left, right

7. Finger analysis mode
- Command:
  - ./app --analyze-fingers
- Displays:
  - per-finger angle
  - per-finger ratio
  - palm orientation
  - FPS
  - ROI status
- Key k appends row to logs/finger_analysis.csv.
- Analysis mode records raw statistics only (no finger-state classification and no threshold decisions).

8. Validation tooling
- Audit/preview now validates:
  - RGB-feature pairing integrity
  - per-class counts
  - missing pairs
  - finger-state distribution
  - palm-direction distribution
  - invalid feature dimension count
- Outputs:
  - logs/dataset_validation.json
  - logs/dataset_validation.md
  - logs/dataset_preview.jpg

## Commands (Step 1)

1. Capture
./app --gesture A --target 1200

2. Two-hand capture
No two-hand command classes in current 26-class setup.

3. Validation
./app --audit --target 1200
./app --preview --target 1200

4. Finger analysis
./app --analyze-fingers

## Notes
- This update intentionally does not modify Step 2 training scripts.
- Existing Step 2 artifacts/config assumptions may require refresh if they previously depended on old Step 1 file paths or feature dimensions.
