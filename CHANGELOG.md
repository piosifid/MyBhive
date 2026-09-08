# Changelog

All notable changes to the b-hive machine learning framework will be documented in this file.

---



## [1.1.0] - AN Status, LZ4 Integration & Advanced Features - 2025

Details of this version are described in the CMS Analysis Note [coming soon]().

#### Added
- **Classifier base class**: New base class [`Classifier_base`](utils/models/base_model.py) and abstract base class [`Classifier`](utils/models/abstract_base_models.py) to standardize the b-hive API and improve model consistency
- **torch.compile Support**: PyTorch 2.0 compilation for improved training performance (use `--use-torch-compile True`)
- **LZ4 data format**: Full support for LZ4-compressed data in `DatasetConstructorTask` for smaller file sizes and faster I/O
- **UParTv2 Model**: Implementation of UParT version 2 model
- **Lost Tracks inputs**: New "lost tracks" (lt) feature category for improved jet characterization
- **Adversarial Attacks**: Framework for testing model robustness (PGD, FGSM, JetFool, Optimizer-based attacks) in the [`adversarial_attacks`](utils/adversarial_attacks) module

#### Changed
- **Data Format**: LZ4 compression is now the default format for dataset processing (previously NumPy)
- **Model Interface**: All models now inherit from standardized base classes for consistent API

---

## [1.0.0] - Initial Release - 2023-12

The initial release version for the b-hive machine learning framework documented in DP note [CMS-DP-2024-020](https://cds.cern.ch/record/2896100?ln=de).

#### Added
- **Core Framework**: Law-based task orchestration system
- **Data Processing**: `DatasetConstructorTask` with `coffea_processor` for converting ROOT files to `numpy structured arrays`
- **Training Pipeline**: `TrainingTask` with reweighting and multi-epoch support
- **Inference**: `InferenceTask` for model evaluation on test datasets
- **Evaluation**: `ROCCurveTask` and `WorkingPointTask` for performance assessment
- **Model Architectures**: DeepJet, ParticleNet, ParticleTransformer implementations
- **Configuration System**: YAML-based configs for HLT, offline (and L1T) scenarios
- **Coffea Processors**: ROOT file processors for NanoAOD and DeepNTuple formats
- **Reweighting**: pT/η-based sample reweighting for balanced training


---
---

# How to Write Changelog Entries

When adding a new entry, follow these guidelines:

**What to Include:**
- **New Features**: New models, tasks, utilities, or capabilities
- **Behavior Changes**: Modifications to existing functionality that users should know about
- **Breaking Changes**: Changes that require user action (config updates, API changes, etc.)
- **Moved/Reorganized**: Files or modules relocated to different directories
- **Performance**: Significant speed or memory improvements
- **Bug Fixes**: Only notable fixes that affect user workflows

**What to Exclude:**
- Internal refactoring that doesn't affect users
- Minor code cleanup or formatting
- Documentation typos (unless they fix critical misunderstandings)

**Format:**
Use the following categories as appropriate:
- `#### Added` - New features, models, tasks
- `#### Changed` - Behavior modifications, moved files
- `#### Deprecated` - Features that will be removed in future versions
- `#### Removed` - Features that have been removed
- `#### Fixed` - Bug fixes
- `#### Performance` - Speed or memory improvements
- `#### Breaking Changes` - Changes requiring user action

---

## Template for Future Entries

```markdown
## [X.Y.Z] - Short Description - YYYY-MM

#### Added
- **Feature Name**: Brief description of what was added and how to use it

#### Changed
- **Component Name**: What changed and how it affects users
- **File Location**: If files moved, note old → new location

#### Breaking Changes
- **What broke**: Clear description of what users need to update

#### Fixed
- **Issue**: What bug was fixed and its impact

#### Performance
- **Optimization**: What was improved and expected speedup
```

