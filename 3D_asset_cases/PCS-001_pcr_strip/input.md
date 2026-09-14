# pcr strip benchmark Input Specification

- `benchmark_id`: PCS-001
- `asset_class`: pcr_strip
- `specification_version`: 1.2.0
- `language`: en

> This Markdown file contains the complete input specification carried over from the prior structured representation. Known values, unknown values, provenance classes, tolerances, and measurement plans are unchanged.

## Asset Identity

- `name`: pcr strip
- `name_en`: pcr_strip
- `representation_mode`: Reproduce the selected configuration where facts are known; do not invent missing device geometry.
- `manufacturer`: Thermo Scientific
- `model`: AB0451
- `configuration`: One eight-tube strip and its separate eight-domed-cap strip; natural polypropylene.

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

- `working_capacity_per_tube`:
  - `id`: DIM-WORKING_CAPACITY_PER_TUBE
  - `value`: `0.2`
  - `unit`: mL
  - `measurement_object`: AB0451
  - `measurement_location`: Each tube
  - `measurement_state`: Unloaded selected configuration
  - `source_class`: M
  - `source_refs`:
    - SRC-01
  - `unknown_reason`: `null` (unknown or not applicable as stated by the adjacent fields)
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
- `closed_capacity_per_tube`:
  - `id`: DIM-CLOSED_CAPACITY_PER_TUBE
  - `value`: `0.25`
  - `unit`: mL
  - `measurement_object`: AB0451
  - `measurement_location`: Each capped tube
  - `measurement_state`: Unloaded selected configuration
  - `source_class`: M
  - `source_refs`:
    - SRC-01
  - `unknown_reason`: `null` (unknown or not applicable as stated by the adjacent fields)
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
- `pitch`:
  - `id`: DIM-PITCH
  - `value`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `unit`: mm
  - `measurement_object`: AB0451
  - `measurement_location`: pitch
  - `measurement_state`: Requires independent reference measurement
  - `source_class`: U
  - `source_refs`:
    - *(none)*
  - `unknown_reason`: Tube center pitch and strip envelope not established by the read source; do not infer them from a generic96-well format.
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
- `cap_interference`:
  - `id`: DIM-CAP_INTERFERENCE
  - `value`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `unit`: mm
  - `measurement_object`: AB0451
  - `measurement_location`: cap_interference
  - `measurement_state`: Requires independent reference measurement
  - `source_class`: U
  - `source_refs`:
    - *(none)*
  - `unknown_reason`: Cap seat profile and interference unknown.
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

## Required Components

- **Item 1 — `CMP-0`**
  - `id`: CMP-0
  - `name`: Connected eight-tube strip
  - `quantity`: `1`
  - `kind`: fixed
  - `parent`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 2 — `CMP-1`**
  - `id`: CMP-1
  - `name`: Eight internal cavities
  - `quantity`: `8`
  - `kind`: cavity
  - `parent`: CMP-0
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 3 — `CMP-2`**
  - `id`: CMP-2
  - `name`: Separate domed cap strip
  - `quantity`: `1`
  - `kind`: removable
  - `parent`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 4 — `CMP-3`**
  - `id`: CMP-3
  - `name`: Domed caps
  - `quantity`: `8`
  - `kind`: fixed
  - `parent`: CMP-2
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01

## Interfaces

- **Item 1 — `IF-0`**
  - `id`: IF-0
  - `description`: Eight positions align together with a compatible0.2 mL block; compatibility is conditional on independently verified geometry.
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 2 — `IF-1`**
  - `id`: IF-1
  - `description`: The matching cap strip closes all eight mouths; removal is a free-body operation,not eight invented hinges.
  - `source_class`: M
  - `source_refs`:
    - SRC-01

## Reference Consumables

- **Item 1 — `REF-0`**
  - `id`: REF-0
  - `configuration`: AB0451 tube strip and supplied matching cap strip; independently matched0.2 mL block geometry
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
  - `description`: Keep all8 cavities distinct and accessible.
  - `source_class`: B
  - `source_refs`:
    - *(none)*
- **Item 2 — `FN-1`**
  - `id`: FN-1
  - `description`: Insert and remove the whole strip; seat each matching cap without lateral mismatch.
  - `source_class`: B
  - `source_refs`:
    - *(none)*

## Protocol Conditioned Requirements

- **Item 1 — `PRO-0`**
  - `id`: PRO-0
  - `description`: B preparation of intact eight-tube PCR strip; cutting capability is excluded.
  - `source_class`: B
  - `source_refs`:
    - *(none)*
  - `preconditions`: Strip supported and uncapped.
  - `action`: Visit each mouth; place cap strip; remove it before retrieving tubes.
  - `expected_postcondition`: All8 sites remain individually addressable.
  - `forbidden_states`:
    - Treating package well count96 as one strip
    - Running with any cap unseated
  - `parameters`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `source_defined_step`:
    - `status`: not_separately_extracted
    - `source_class`: U
    - `source_refs`:
      - *(none)*
    - `statement`: `null` (unknown or not applicable as stated by the adjacent fields)
    - `source_note`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `benchmark_test_action`:
    - `source_class`: B
    - `action`: Visit each mouth; place cap strip; remove it before retrieving tubes.
    - `parameters`: `null` (unknown or not applicable as stated by the adjacent fields)
    - `reason`: Benchmark-authored observable action derived from the cited source context; it is not represented as a verbatim manufacturer, protocol, or standards requirement.

## Source Documents

- **Item 1 — `SRC-01`**
  - `id`: SRC-01
  - `title`: Tubes and Domed Caps,strips of8
  - `publisher`: Thermo Scientific
  - `url`: https://www.thermofisher.com/order/catalog/product/AB0451
  - `version_or_publication_date`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `locator`: Description,Features and Specifications;8-strip product identity takes precedence over ambiguous No.of Wells96 field.
  - `access_date`: 2026-09-08
  - `verification_status`: body_read
  - `supports_requirement_ids`:
    - DIM-WORKING_CAPACITY_PER_TUBE
    - DIM-CLOSED_CAPACITY_PER_TUBE
    - CMP-0
    - CMP-1
    - CMP-2
    - CMP-3
    - IF-0
    - IF-1
    - REF-0
    - REQ-VIS

## Scope And Assumptions

- `included`:
  - Geometry and rigid-body contact
  - Applicable articulation and device-state logic
- `excluded`:
  - Real fluid flow or delivered volume
  - Heat transfer and experimental efficacy
  - Independent extraction of requirements from raw sources
  - Cutting strip links
  - PCR and vapor sealing
- `runtime_dependencies`:
  - Submitted asset and pinned MuJoCo environment
  - Independent geometry, contact and state checkers
  - Assigned human visual reviewer
  - Independent reference model: AB0451 tube strip and supplied matching cap strip; independently matched0.2 mL block geometry
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
    - `initial`: Uncapped strip supported in independently verified block.
    - `action`: Visit each of8 mouths with1 mm probe; descend3 mm and retract10 mm at10 mm/s.
    - `observable`: Cavity connectivity and probe clearance at each index.
    - `pass_condition`: Exactly8 separate cavities; every mouth reachable; no cross-cavity solid bridge or unintended overlap >0.2 mm.
    - `source_class`: B
    - `note`: Proxy acceptance conditions,not manufacturer tolerances; source requirements retain their own provenance.
  - `T-CASE-1`:
    - `initial`: Cap strip30 mm above seated tubes; independent references loaded.
    - `action`: Lower along-Z to measured seat; lift30 mm; lift tube strip30 mm.
    - `observable`: Eight seat residuals and tube release clearance.
    - `pass_condition`: All8 seats within0.2 mm of independent reference contact positions; complete strip releases without tube detachment or collision.
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
    - `condition`: Exactly8 separate cavities; every mouth reachable; no cross-cavity solid bridge or unintended overlap >0.2 mm.
    - `target_path`: input.test_conditions.asset_specific.T-CASE-0
    - `source_class`: B
  - `T-CASE-1`:
    - `condition`: All8 seats within0.2 mm of independent reference contact positions; complete strip releases without tube detachment or collision.
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
  - `locator`: Description,Features and Specifications;8-strip product identity takes precedence over ambiguous No.of Wells96 field.
  - `url`: https://www.thermofisher.com/order/catalog/product/AB0451
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
