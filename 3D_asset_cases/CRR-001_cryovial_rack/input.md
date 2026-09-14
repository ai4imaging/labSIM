# cryovial rack benchmark Input Specification

- `benchmark_id`: CRR-001
- `asset_class`: cryovial_rack
- `specification_version`: 1.2.0
- `language`: en

> This Markdown file contains the complete input specification carried over from the prior structured representation. Known values, unknown values, provenance classes, tolerances, and measurement plans are unchanged.

## Asset Identity

- `name`: cryovial rack
- `name_en`: cryovial_rack
- `representation_mode`: Reproduce the selected configuration where facts are known; do not invent missing device geometry.
- `manufacturer`: Thermo Scientific Nunc
- `model`: 376589
- `configuration`: PPO40-position4x10 interlocking cryotube holder.

## Task Instruction

- `id`: REQ-ASSET
- `description`: Future deliverable: compilable MuJoCo MJCF with relative mesh resources and locatable component, joint and interaction mappings. This task delivers specifications only.
- `source_class`: B
- `source_refs`:
  - *(none)*
- `output_format`: MJCF + meshes + semantic mapping
- `required_capabilities`:
  - *(none)*

## Dimensions

- `envelope`:
  - `id`: DIM-ENVELOPE
  - `value`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `unit`: mm
  - `measurement_object`: 376589
  - `measurement_location`: envelope
  - `measurement_state`: Requires independent reference measurement
  - `source_class`: U
  - `source_refs`:
    - *(none)*
  - `unknown_reason`: Overall dimensions and hole pitch not published in read page.
  - `tolerance`:
    - `source_class`: B
    - `full_relative_error_max`: `0.05`
    - `partial_relative_error_max`: `0.1`
    - `reason`: Initial geometric reproduction tolerance; not empirically calibrated. This is not a manufacturer tolerance.
    - `is_manufacturer_tolerance`: `false`
  - `required_for_spec_completion`: `true`
  - `critical`: `true`
  - `hard_fail_relative_error_max`:
    - `value`: `0.2`
    - `source_class`: B
    - `reason`: Benchmark hard-failure threshold for a critical dimension; not a manufacturer tolerance.
  - `measurement_plan`:
    - `status`: required_not_performed
    - `source_class`: B
    - `method`: Measure the stated feature on three physical units with three repetitions per unit, using a sectioned sample or non-contact metrology where the feature is not externally accessible.
    - `instrument`: Calibrated digital caliper, micrometer, or coordinate measurement system selected for the feature
    - `sample_count`: `3`
    - `repetitions_per_sample`: `3`
    - `required_traceability`:
      - instrument identifier and calibration certificate
      - sample catalog number and lot when available
      - operator, date, raw readings, summary statistic and measurement uncertainty
      - photograph or drawing showing the measurement datum and asset state
    - `value_population_rule`: Keep value null and source_class U until the raw record and calibration evidence are reviewed; then record the statistic and retain the measurement record reference.
- `key_geometry`:
  - `id`: DIM-KEY_GEOMETRY
  - `value`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `unit`: mm
  - `measurement_object`: 376589
  - `measurement_location`: key_geometry
  - `measurement_state`: Requires independent reference measurement
  - `source_class`: U
  - `source_refs`:
    - *(none)*
  - `unknown_reason`: Anti-rotation grip and interlocking-edge geometry require independent measurement.
  - `tolerance`:
    - `source_class`: B
    - `full_relative_error_max`: `0.05`
    - `partial_relative_error_max`: `0.1`
    - `reason`: Initial geometric reproduction tolerance; not empirically calibrated. This is not a manufacturer tolerance.
    - `is_manufacturer_tolerance`: `false`
  - `required_for_spec_completion`: `true`
  - `critical`: `true`
  - `hard_fail_relative_error_max`:
    - `value`: `0.2`
    - `source_class`: B
    - `reason`: Benchmark hard-failure threshold for a critical dimension; not a manufacturer tolerance.
  - `measurement_plan`:
    - `status`: required_not_performed
    - `source_class`: B
    - `method`: Measure the stated feature on three physical units with three repetitions per unit, using a sectioned sample or non-contact metrology where the feature is not externally accessible.
    - `instrument`: Calibrated digital caliper, micrometer, or coordinate measurement system selected for the feature
    - `sample_count`: `3`
    - `repetitions_per_sample`: `3`
    - `required_traceability`:
      - instrument identifier and calibration certificate
      - sample catalog number and lot when available
      - operator, date, raw readings, summary statistic and measurement uncertainty
      - photograph or drawing showing the measurement datum and asset state
    - `value_population_rule`: Keep value null and source_class U until the raw record and calibration evidence are reviewed; then record the statistic and retain the measurement record reference.
- `external_long_side`:
  - `id`: DIM-LONG-SIDE
  - `value`: `202`
  - `unit`: mm
  - `measurement_object`: Nunc 376589 rack
  - `measurement_location`: Longer external planar side; source does not provide height
  - `measurement_state`: Unloaded selected configuration
  - `source_class`: M
  - `source_refs`:
    - SRC-02
  - `unknown_reason`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `tolerance`:
    - `source_class`: B
    - `full_relative_error_max`: `0.05`
    - `partial_relative_error_max`: `0.1`
    - `reason`: Initial benchmark reproduction tolerance; not empirically calibrated and not a manufacturer tolerance.
    - `is_manufacturer_tolerance`: `false`
  - `required_for_spec_completion`: `true`
  - `critical`: `true`
  - `hard_fail_relative_error_max`:
    - `value`: `0.2`
    - `source_class`: B
    - `reason`: Benchmark hard-failure threshold for a critical dimension; not a manufacturer tolerance.
- `external_short_side`:
  - `id`: DIM-SHORT-SIDE
  - `value`: `102`
  - `unit`: mm
  - `measurement_object`: Nunc 376589 rack
  - `measurement_location`: Shorter external planar side; source does not provide height
  - `measurement_state`: Unloaded selected configuration
  - `source_class`: M
  - `source_refs`:
    - SRC-02
  - `unknown_reason`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `tolerance`:
    - `source_class`: B
    - `full_relative_error_max`: `0.05`
    - `partial_relative_error_max`: `0.1`
    - `reason`: Initial benchmark reproduction tolerance; not empirically calibrated and not a manufacturer tolerance.
    - `is_manufacturer_tolerance`: `false`
  - `required_for_spec_completion`: `true`
  - `critical`: `true`
  - `hard_fail_relative_error_max`:
    - `value`: `0.2`
    - `source_class`: B
    - `reason`: Benchmark hard-failure threshold for a critical dimension; not a manufacturer tolerance.

## Required Components

- **Item 1 — `CMP-0`**
  - `id`: CMP-0
  - `name`: Holder body
  - `quantity`: `1`
  - `kind`: fixed
  - `parent`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 2 — `CMP-1`**
  - `id`: CMP-1
  - `name`: Tube positions
  - `quantity`: `40`
  - `kind`: cavity
  - `parent`: CMP-0
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 3 — `CMP-2`**
  - `id`: CMP-2
  - `name`: Interlocking edges
  - `quantity`: `1`
  - `kind`: fixed
  - `parent`: CMP-0
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 4 — `CMP-3`**
  - `id`: CMP-3
  - `name`: Tube gripping features
  - `quantity`: `1`
  - `kind`: fixed
  - `parent`: CMP-0
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01

## Interfaces

- **Item 1 — `IF-0`**
  - `id`: IF-0
  - `description`: Compatible cryotubes engage the anti-rotation grip for one-handed cap manipulation.
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 2 — `IF-1`**
  - `id`: IF-1
  - `description`: Rack edges interlock with another identical holder;key geometry is not yet verified.
  - `source_class`: M
  - `source_refs`:
    - SRC-01

## Reference Consumables

- **Item 1 — `REF-0`**
  - `id`: REF-0
  - `configuration`: Nunc375418 candidate cryotube and376589 holder independent models;exact key fit verification required
  - `source_refs`:
    - SRC-01
  - `source_class`: M
  - `model_status`: missing
  - `model_version`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `path`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `sha256`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `frozen`: `false`

## Articulation Requirements

- `id`: REQ-JOINT
- `applicable`: `false`
- `reason`: No internal moving mechanism. Free placement in the world is not an internal joint.
- `source_class`: B
- `source_refs`:
  - *(none)*
- `joints`:
  - *(none)*

## Functional Requirements

- **Item 1 — `FN-0`**
  - `id`: FN-0
  - `description`: Retain40 cryotubes and resist cap-axis torque.
  - `source_class`: B
  - `source_refs`:
    - *(none)*
- **Item 2 — `FN-1`**
  - `id`: FN-1
  - `description`: Interlock two holders without obstructing outer tubes.
  - `source_class`: B
  - `source_refs`:
    - *(none)*

## Protocol Conditioned Requirements

- **Item 1 — `PRO-0`**
  - `id`: PRO-0
  - `description`: One-handed capping and decapping are documented holder uses.
  - `source_class`: M
  - `source_refs`:
    - SRC-01
  - `preconditions`: Tube engaged with gripping feature.
  - `action`: Manipulate cap while holder restrains tube.
  - `expected_postcondition`: Tube remains supported.
  - `forbidden_states`:
    - Treating tube rotation with cap as successful restraint
  - `parameters`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `source_defined_step`:
    - `status`: extracted
    - `source_class`: M
    - `source_refs`:
      - SRC-01
    - `statement`: One-handed capping and decapping are documented holder uses.
    - `source_note`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `benchmark_test_action`:
    - `source_class`: B
    - `action`: Manipulate cap while holder restrains tube.
    - `parameters`: `null` (unknown or not applicable as stated by the adjacent fields)
    - `reason`: Benchmark-authored observable action derived from the cited source context; it is not represented as a verbatim manufacturer, protocol, or standards requirement.

## Source Documents

- **Item 1 — `SRC-01`**
  - `id`: SRC-01
  - `title`: CryoTube Holders
  - `publisher`: Thermo Scientific Nunc
  - `url`: https://www.thermofisher.com/order/catalog/product/376589
  - `version_or_publication_date`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `locator`: Selected376589 specification table and description
  - `access_date`: 2026-09-08
  - `verification_status`: body_read
  - `supports_requirement_ids`:
    - CMP-0
    - CMP-1
    - CMP-2
    - CMP-3
    - IF-0
    - IF-1
    - REF-0
    - PRO-0
    - REQ-VIS
- **Item 2 — `SRC-02`**
  - `id`: SRC-02
  - `title`: Thermo Scientific Nunc Product Catalog - CryoTube Rack
  - `url`: https://assets.thermofisher.com/TFS-Assets/LSG/brochures/Nunc%E4%BA%A7%E5%93%81%E7%9B%AE%E5%BD%95.pdf
  - `version`: catalog edition available online in 2026
  - `locator`: Catalog page 47, item 376589: external dimensions 202 x 102 mm, 40 positions, PPO material and starfoot one-hand operation.
  - `access_date`: 2026-09-09
  - `verification_status`: body_read
  - `supports_requirement_ids`:
    - DIM-LONG-SIDE
    - DIM-SHORT-SIDE
    - IF-0
    - FN-0

## Scope And Assumptions

- `included`:
  - Geometry and rigid-body contact
  - Applicable articulation and device-state logic
- `excluded`:
  - Real fluid flow or delivered volume
  - Heat transfer and experimental efficacy
  - Independent extraction of requirements from raw sources
- `runtime_dependencies`:
  - Submitted asset and pinned MuJoCo environment
  - Independent geometry, contact and state checkers
  - Assigned human visual reviewer
  - Independent reference model: Nunc375418 candidate cryotube and376589 holder independent models;exact key fit verification required
  - Pinned and independently justified mass,inertia and contact parameters
  - Independent source-truth geometry for dimensions marked unknown
- `certification_blocked_until_dependencies_resolved`: `true`

## Test Conditions

- `id`: REQ-TEST
- `source_class`: B
- `source_refs`:
  - *(none)*
- `gravity_m_s2`:
  - `0`
  - `0`
  - `-9.81`
- `time_step_s`: `0.001`
- `contact_solver`: Pin and record MuJoCo version and solver settings before running; missing configuration gives invalid_test.
- `linear_speed_mm_s`: `10`
- `angular_speed_deg_s`: `30`
- `repetitions`: `3`
- `timeout_s`: `30`
- `abnormal_penetration_max_mm`: `0.2`
- `stable_translation_max_mm`: `1`
- `stable_tilt_max_deg`: `2`
- `reason`: Initial quasi-static proxy conditions, not manufacturer operating speeds or verified material properties. This is not a manufacturer tolerance.
- `asset_specific`:
  - `T-CASE-0`:
    - `initial`: All40 slots occupied with verified reference tubes.
    - `action`: Release5 s;apply0.005 N m about each tube axis for2 s.
    - `observable`: Axial drift and angular displacement.
    - `pass_condition`: 40/40 retained;drift <=1 mm;rotation <=2 deg;overlap <=0.2 mm.
    - `source_class`: B
    - `note`: Proxy acceptance conditions,not manufacturer tolerances; source requirements retain their own provenance.
  - `T-CASE-1`:
    - `initial`: Two empty identical holders20 mm apart.
    - `action`: Bring mating edges to reference engagement;apply0.5 N lateral load for2 s;separate.
    - `observable`: Edge fit and relative displacement.
    - `pass_condition`: Interlock reaches reference pose within0.2 mm;relative displacement <=1 mm under load;reversible without breakage proxy.
    - `source_class`: B
    - `note`: Proxy acceptance conditions,not manufacturer tolerances; source requirements retain their own provenance.
- `acceptance_by_check`:
  - `T-LOAD`:
    - `condition`: No errors, resources resolved and all states finite.
    - `target_path`: input.task_instruction
    - `source_class`: B
  - `T-STRUCT`:
    - `condition`: Every critical component and connection conforms.
    - `target_path`: input.required_components
    - `source_class`: B
  - `T-VIS`:
    - `condition`: Every listed feature is identifiable.
    - `target_path`: input.visual_requirements
    - `source_class`: B
  - `T-CASE-0`:
    - `condition`: 40/40 retained;drift <=1 mm;rotation <=2 deg;overlap <=0.2 mm.
    - `target_path`: input.test_conditions.asset_specific.T-CASE-0
    - `source_class`: B
  - `T-CASE-1`:
    - `condition`: Interlock reaches reference pose within0.2 mm;relative displacement <=1 mm under load;reversible without breakage proxy.
    - `target_path`: input.test_conditions.asset_specific.T-CASE-1
    - `source_class`: B
  - `T-DIM`:
    - `condition`: All required dimension errors <=10%;unknown truth blocks evaluation.
    - `target_path`: input.dimensions
    - `source_class`: B
- `is_manufacturer_tolerance`: `false`

## Visual Requirements

- `id`: REQ-VIS
- `description`: Observable features in fixed views.
- `source_class`: M
- `source_refs`:
  - SRC-01
- `reference`:
  - `source_id`: SRC-01
  - `locator`: Selected376589 specification table and description
  - `url`: https://www.thermofisher.com/order/catalog/product/376589
  - `verification_status`: Source text read; image-specific appearance review remains a runtime dependency.
- `views`:
  - front
  - left
  - top
  - front_left_45deg
- `state`: Upright and stationary; also open-cover views where applicable.
- `features`:
  - Material appearance and silhouette consistent with the source image
- `reviewer_status`: not_assigned
- `note`: Family illustrations support structural features only, not exact pixel dimensions of the selected SKU.
- `review_protocol`:
  - `source_class`: B
  - `render_resolution_px`:
    - `width`: `1600`
    - `height`: `1600`
  - `projection`: orthographic for named orthographic views; perspective only for the named 45-degree view
  - `background`: neutral mid-gray, uniform illumination, no depth-of-field blur
  - `asset_state`: Upright and stationary; also open-cover views where applicable.
  - `occlusion_policy`: Generate an additional unobstructed view when a required feature is hidden; a hidden feature is not automatically conforming.
  - `reviewer_policy`:
    - `mode`: two_independent_human_reviewers_or_one_pinned_vision_model
    - `human_requirement`: Record two reviewer IDs and resolve disagreements before certification.
    - `model_requirement`: Record provider, model/version, prompt hash and image hash. Unpinned or unavailable models yield blocked_dependency.
    - `current_assignment`: `null` (unknown or not applicable as stated by the adjacent fields)
    - `status`: blocked_dependency
