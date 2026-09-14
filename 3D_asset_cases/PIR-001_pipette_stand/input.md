# pipette stand benchmark Input Specification

- `benchmark_id`: PIR-001
- `asset_class`: pipette_stand
- `specification_version`: 1.2.0
- `language`: en

> This Markdown file contains the complete input specification carried over from the prior structured representation. Known values, unknown values, provenance classes, tolerances, and measurement plans are unchanged.

## Asset Identity

- `name`: pipette stand
- `name_en`: pipette_stand
- `representation_mode`: Reproduce the selected configuration where facts are known; do not invent missing device geometry.
- `manufacturer`: Eppendorf
- `model`: Pipette Carousel2 3116000015
- `configuration`: Six noncharging holders for Research plus family;not whiteResearch3-only3116000236.

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
  - `measurement_object`: Pipette Carousel2 3116000015
  - `measurement_location`: envelope
  - `measurement_state`: Requires independent reference measurement
  - `source_class`: U
  - `source_refs`:
    - *(none)*
  - `unknown_reason`: Base diameter,height and holder spacing unavailable.
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
- `hanger_geometry`:
  - `id`: DIM-HANGER_GEOMETRY
  - `value`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `unit`: mm
  - `measurement_object`: Pipette Carousel2 3116000015
  - `measurement_location`: hanger_geometry
  - `measurement_state`: Requires independent reference measurement
  - `source_class`: U
  - `source_refs`:
    - *(none)*
  - `unknown_reason`: Holder engagement and suspended-tip clearance need independent geometry.
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
- `reference_tip_length`:
  - `id`: DIM-REF-TIP-L
  - `value`: `53`
  - `unit`: mm
  - `measurement_object`: Yellow 2-200 uL epT.I.P.S. on a compatible Research plus pipette
  - `measurement_location`: Tip end-to-end length used to define minimum suspended-tip clearance; not a carousel envelope dimension
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
  - `name`: Base with rubber feet
  - `quantity`: `1`
  - `kind`: fixed
  - `parent`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 2 — `CMP-1`**
  - `id`: CMP-1
  - `name`: Rotating carousel
  - `quantity`: `1`
  - `kind`: moving
  - `parent`: CMP-0
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 3 — `CMP-2`**
  - `id`: CMP-2
  - `name`: Research plus holders
  - `quantity`: `6`
  - `kind`: fixed
  - `parent`: CMP-1
  - `critical`: `true`
  - `source_class`: M
  - `source_refs`:
    - SRC-01

## Interfaces

- **Item 1 — `IF-0`**
  - `id`: IF-0
  - `description`: Six matching holders support Research plus pipettes;no charging contacts required.
  - `source_class`: M
  - `source_refs`:
    - SRC-01
- **Item 2 — `IF-1`**
  - `id`: IF-1
  - `description`: Interchangeable holders must be the selected Research plus type.
  - `source_class`: M
  - `source_refs`:
    - SRC-01

## Reference Consumables

- **Item 1 — `REF-0`**
  - `id`: REF-0
  - `configuration`: Six Research plus3120000054 pipettes without tips;independent holder/carousel geometry
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
- `applicable`: `true`
- `reason`: The selected configuration has moving mechanisms.
- `source_class`: B
- `source_refs`:
  - *(none)*
- `joints`:
  - **Item 1 — `J-0`**
    - `id`: J-0
    - `parent_component`: CMP-0
    - `child_component`: CMP-1
    - `type`: continuous_revolute
    - `coordinate_frame`: Local body frame; +Z upright. Axis location must come from independent reference geometry.
    - `axis`:
      - `0`
      - `0`
      - `1`
    - `zero`: Holder1 faces+X
    - `range`:
      - `min`: `null` (unknown or not applicable as stated by the adjacent fields)
      - `max`: `null` (unknown or not applicable as stated by the adjacent fields)
      - `unit`: deg
    - `range_source_class`: B
    - `limits`: Continuous carousel rotation;Btest360 deg.
    - `locking_conditions`: None
    - `source_class`: B
    - `source_refs`:
      - *(none)*
    - `unknown_physical_geometry`: Pivot or sliding guide geometry requires independent reference.

## Functional Requirements

- **Item 1 — `FN-0`**
  - `id`: FN-0
  - `description`: Suspend pipettes without contacting bench or actuating plungers.
  - `source_class`: B
  - `source_refs`:
    - *(none)*
- **Item 2 — `FN-1`**
  - `id`: FN-1
  - `description`: Rotate loaded carousel to present each holder.
  - `source_class`: B
  - `source_refs`:
    - *(none)*

## Protocol Conditioned Requirements

- **Item 1 — `PRO-0`**
  - `id`: PRO-0
  - `description`: B storage/retrieval fragment based on documented holder compatibility.
  - `source_class`: B
  - `source_refs`:
    - *(none)*
  - `preconditions`: Pipettes without tips;carousel stopped.
  - `action`: Hang by proper holder contact;rotate to chosen position;remove.
  - `expected_postcondition`: Remaining pipettes stay suspended.
  - `forbidden_states`:
    - Supporting by fragile tip cone
    - Charging claim for mechanical carousel
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
    - `action`: Hang by proper holder contact;rotate to chosen position;remove.
    - `parameters`: `null` (unknown or not applicable as stated by the adjacent fields)
    - `reason`: Benchmark-authored observable action derived from the cited source context; it is not represented as a verbatim manufacturer, protocol, or standards requirement.

## Source Documents

- **Item 1 — `SRC-01`**
  - `id`: SRC-01
  - `title`: Pipette Holder System
  - `publisher`: Eppendorf
  - `url`: https://www.eppendorf.com/gb-en/Products/Liquid-Handling/Dispenser-Pipette-Accessories/Eppendorf-Pipette-Holder-System-p-PF-218982
  - `version_or_publication_date`: `null` (unknown or not applicable as stated by the adjacent fields)
  - `locator`: 3116000015 row and Features
  - `access_date`: 2026-09-08
  - `verification_status`: body_read
  - `supports_requirement_ids`:
    - CMP-0
    - CMP-1
    - CMP-2
    - IF-0
    - IF-1
    - REF-0
    - REQ-VIS
- **Item 2 — `SRC-02`**
  - `id`: SRC-02
  - `title`: Eppendorf Research plus Operating Manual
  - `url`: https://www.eppendorf.com/product-media/doc/en/174967/Eppendorf_Liquid-Handling_Operating-manual_Research-plus_Eppendorf-Research-plus.pdf
  - `version`: manufacturer operating manual available in 2026
  - `locator`: Tip allocation table: yellow 2-200 uL epT.I.P.S. length 53 mm; used as a fixed suspended-clearance reference, not as a carousel dimension.
  - `access_date`: 2026-09-09
  - `verification_status`: body_read
  - `supports_requirement_ids`:
    - DIM-REF-TIP-L
    - IF-0

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
  - Independent reference model: Six Research plus3120000054 pipettes without tips;independent holder/carousel geometry
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
    - `initial`: Six references hung;base on bench.
    - `action`: Rotate360 deg at30 deg/s;stop at60 deg intervals.
    - `observable`: Angle and mutual clearance.
    - `pass_condition`: Angle error <=2 deg;no pipette collisions or unintended overlaps >0.2 mm.
    - `source_class`: B
    - `note`: Proxy acceptance conditions,not manufacturer tolerances; source requirements retain their own provenance.
  - `T-CASE-1`:
    - `initial`: Fully loaded carousel.
    - `action`: Release5 s;lift each pipette30 mm along verified removal path and replace.
    - `observable`: Tip-to-bench clearance and neighbor stability.
    - `pass_condition`: All6 remain supported;tip cones clear bench byB>=5 mm;neighbor drift <=1 mm;no plunger actuation.
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
    - `condition`: Angle error <=2 deg;no pipette collisions or unintended overlaps >0.2 mm.
    - `target_path`: input.test_conditions.asset_specific.T-CASE-0
    - `source_class`: B
  - `T-CASE-1`:
    - `condition`: All6 remain supported;tip cones clear bench byB>=5 mm;neighbor drift <=1 mm;no plunger actuation.
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
  - `locator`: 3116000015 row and Features
  - `url`: https://www.eppendorf.com/gb-en/Products/Liquid-Handling/Dispenser-Pipette-Accessories/Eppendorf-Pipette-Holder-System-p-PF-218982
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
