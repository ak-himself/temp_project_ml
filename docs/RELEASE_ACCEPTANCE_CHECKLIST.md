# Release Acceptance Checklist — ASL Static Gesture Interpreter

Related Jira work item: ASGI2-37

## Purpose

This checklist defines the evidence to collect before treating ASL Static Gesture Interpreter v1.0 as release-ready. A checked box means the corresponding verification has actually been performed and recorded. The existence of this checklist does not mean that every release gate has passed.

## 1. Repository and build readiness

- [ ] Confirm the target GitHub repository and default branch are correct.
- [ ] Confirm Jira's Development panel displays the connected repository.
- [ ] Install dependencies from a clean supported Python 3.12.8 environment:

  ```bash
  python -m pip install -r requirements.txt
  ```

- [ ] Run the CLI sanity check:

  ```bash
  python app.py --help
  ```

- [ ] Confirm required model artifacts are present and loadable.
- [ ] Record the exact source commit and model-artifact identifiers used for the release candidate.

## 2. Dataset and feature validation

- [ ] Run the dataset audit and retain its actual output:

  ```bash
  python app.py --audit --target 1200
  ```

- [ ] Run the dataset preview and inspect class coverage and invalid samples:

  ```bash
  python app.py --preview --target 1200
  ```

- [ ] Verify the documented 26-class A–Z mapping.
- [ ] Verify the 21-point MediaPipe landmark pipeline and the expected 119-dimensional feature layout against the checked-out code and artifacts.
- [ ] Record any class imbalance, missing data or invalid samples found.

## 3. Model and inference validation

- [ ] Confirm the XGBoost, Random Forest, MLP and CNN artifacts required by the selected inference mode are available.
- [ ] Confirm the stacking meta-model and configuration are compatible with the base-model artifacts.
- [ ] Verify the chosen CNN input representation against the implementation. The current repository README describes RGB hand crops, while Jira ASGI2-10 specifies a skeleton-image CNN branch; resolve this discrepancy before claiming that requirement is satisfied.
- [ ] Launch the final UI:

  ```bash
  python app.py --ui-final
  ```

- [ ] Exercise representative A–Z gestures and record actual outcomes.
- [ ] Test low-confidence/unknown routing, temporal stability, missing-hand behaviour and camera interruption handling.
- [ ] Record CPU inference measurements using a stated machine, warm-up procedure, sample count and measurement method. Do not substitute an unmeasured target for an observed result.

## 4. Regression and user-interface checks

- [ ] Run the available evaluation/regression scripts.
- [ ] Verify character insertion, clearing and editing behaviour in the live interface.
- [ ] Verify representative visually similar gesture pairs.
- [ ] Verify behaviour under changed orientation and lighting.
- [ ] Record defects, severity, reproduction steps and their Jira issue keys.
- [ ] Re-run affected checks after fixes and retain the actual output.

## 5. GitHub and Jira traceability

- [ ] Confirm the repository is visible under Jira Development, not merely that an organisation connection exists.
- [ ] Confirm at least two relevant Jira issues contain linked branches, commits or pull requests.
- [ ] Confirm each pull request has a review/validation record appropriate to its change.
- [ ] Confirm merged pull requests are shown as merged in GitHub and are surfaced in the corresponding Jira Development panels.
- [ ] Update the team's Definition of Done to require integrated code, suitable tests/evaluation evidence and updated run instructions; use merged-PR evidence where code changes are involved.

## 6. Release scope and version

- [ ] Confirm the v1.0 Fix Version contains only work approved for the release.
- [ ] Confirm every in-scope item has a verified disposition: Done, explicitly deferred, or blocked with an owner and rationale.
- [ ] Confirm unresolved defects and known limitations are documented.
- [ ] Confirm the release notes describe actual delivered functionality and known limitations.
- [ ] Set a release date that matches the real or explicitly reconstructed project record.
- [ ] Mark the version Released only after the release scope and acceptance gates have been verified.

## 7. Release decision record

Complete this section at the actual acceptance review.

| Field | Recorded value |
|---|---|
| Release candidate commit | Pending verification |
| Python/runtime environment | Pending verification |
| Dataset revision | Pending verification |
| Model artifacts | Pending verification |
| Evaluation report | Pending execution/inspection |
| CPU latency measurement | Pending measurement |
| Open critical/high defects | Pending review |
| Jira Fix Version | v1.0 (currently unreleased) |
| Acceptance decision | Pending review |
| Accepted by / date | Pending review |

## Current limitations to resolve

1. Jira currently shows the v1.0 version with three completed data-pipeline stories assigned; it is not yet the full approved release scope.
2. Jira's Development page has shown the GitHub organisation as connected but the repository list as empty. Repository visibility must be verified in the Development panel.
3. The GitHub repository currently has a single `main` branch and no pull requests at the time this checklist was prepared. New PRs created for documentation work must not be presented as historical implementation PRs.
4. Sprint containers are intentionally kept open for the Agile evidence exercise. An open sprint is not evidence that all work in that sprint is complete.
