Full Reproducibility Modules
This directory contains the detailed, script-by-script implementation of the full imaging and machine-learning workflow used in the PD-MSA MRI study.

Scope
The modules in ordered_full_scripts/ cover:

structural feature extraction from cortical and subcortical MRI outputs
FC/SC graph-measure preparation for the multimodal benchmark
discovery-cohort model development
derivation of the locked T1-Top20 signature
external clinical validation with frozen preprocessing assets and locked thresholds
iRBD stratification using the locked disease-signature framework
manuscript figure generation
Recommended Script Order
Run the scripts in the numeric order shown below:

01_01_dcm_to_nifti.sh to 04_02_group_analysis_thickness.py
05_01_discovery_multimodal_benchmark_glmnet.R
05_02_discovery_t1_benchmark_glmnet.R
05_03_discovery_top20_and_irbd_core.R
05_04_train_locked_pd_msa_model_for_external.R
05_05_discovery_t1_top20_multiclass.R
05_06_discovery_msa_subtype_analysis.R
06_01_external_clinical_validation_locked_model.R
07_01_irbd_stratification_locked_thresholds.R
08_01_generate_manuscript_figures.R
Additional plotting or appendix-style scripts can be run afterwards if needed.

Reproducibility Principles
discovery and external analyses are separated
thresholds used in external validation are locked from discovery
frozen ComBat assets are reused for downstream application without refitting
external cohorts are not used for model tuning
AUC confidence intervals use DeLong
non-AUC binary performance intervals use exact binomial confidence intervals
Notes
ordered_full_scripts/_ordered_copy_manifest.csv records the mapping from the original development filenames to the ordered filenames used here.
Some scripts retain legacy development comments, but the current ordered versions are the maintained full-length reproducibility copies.
Machine-specific absolute paths may need to be updated before execution in a new environment.
Minimal Citation Description
Suggested repository description:

Code for multimodal and structural MRI analysis to differentiate Parkinson disease from multiple system atrophy, including discovery, external validation, and iRBD stratification workflows.
