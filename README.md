# MORPHEUS — Short Sleep Prediction with NHANES 2017–2020

MORPHEUS is a serious end-to-end tabular machine learning project focused on predicting short sleep from real-world health survey data built from NHANES 2017–2020.

The project combines structured dataset assembly, disciplined exploratory analysis, feature engineering, model route comparison, recipe selection, threshold-aware refinement, and lightweight interpretation under a reusable workflow philosophy called MARS (Mechanical Automation for Reusable Science).

--------------------------------------------------------------------------------
PROJECT OBJECTIVE
--------------------------------------------------------------------------------

The goal of MORPHEUS is to predict whether an individual belongs to the short sleep group using demographic, anthropometric, cardiovascular, behavioural, and sleep-related variables.

Target definition:
- short_sleep = 1 if sleep_hours_weekday < 7
- short_sleep = 0 if sleep_hours_weekday >= 7

This project explicitly avoids target leakage. The source variable used to define the target, sleep_hours_weekday, is not allowed into modeling.

--------------------------------------------------------------------------------
DATASET
--------------------------------------------------------------------------------

The dataset was assembled from NHANES 2017–2020 and integrates variables from multiple survey modules.

NHANES source modules used:
- Sleep Disorders
- Demographics
- Body Measures
- Blood Pressure – Oscillometric Measurement
- Smoking
- Alcohol Use
- Physical Activity
- Depression Screener

The raw assembly process was completed in 00_dataset_assembly.ipynb, and the modeling workflow starts from the already-built integrated dataset.

--------------------------------------------------------------------------------
WHY THIS PROJECT MATTERS
--------------------------------------------------------------------------------

Sleep duration is connected to broader health, lifestyle, and socioeconomic context. Predicting short sleep is not only a technical binary classification exercise; it is also an opportunity to explore how behavioural, anthropometric, cardiovascular, and derived sleep-pattern signals interact in a realistic health-data setting.

MORPHEUS was designed as a portfolio-grade project with emphasis on:
- methodological discipline
- readable notebook structure
- explicit route comparison
- threshold-aware model selection
- reproducible handoff between stages
- reusable project infrastructure

--------------------------------------------------------------------------------
PROJECT PHILOSOPHY
--------------------------------------------------------------------------------

This repository follows a modular working style:
- notebooks for reasoning, analysis, and interpretation
- src for reusable utilities
- configs for declarative choices
- artifacts for persisted intermediate outputs
- docs for supporting rationale and non-notebook documentation

The broader workflow philosophy is called:

MARS — Mechanical Automation for Reusable Science

MARS is the reusable system behind the project and is based on a simple idea:
- automate the mechanical parts
- preserve human attention for interpretation and decision-making

In practice:
- notebooks should remain clean, technical, and narrative
- repeated mechanical work should move toward reusable code and configuration
- experimental logic should stay explicit and inspectable

--------------------------------------------------------------------------------
REPOSITORY STRUCTURE
--------------------------------------------------------------------------------

morpheus-sleep-health-ml/
├── notebooks/
│   ├── 00_dataset_assembly.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_refinement.ipynb
├── artifacts/
│   └── 03_feature_engineering/
├── docs/
├── src/
├── configs/
└── ...

The exact contents may evolve, but the project is structured around staged notebook handoffs and persisted artifacts.

--------------------------------------------------------------------------------
NOTEBOOK WORKFLOW
--------------------------------------------------------------------------------

00_dataset_assembly.ipynb
Builds the integrated modeling dataset from NHANES source modules.

01_eda.ipynb
Validates the modeling cohort and explores:
- target construction
- missingness
- variable structure
- cohort logic
- general data quality

02_modeling.ipynb
Builds the initial modeling benchmark and shortlist of candidate routes.

This stage includes:
- baselines
- extended benchmark comparison
- route narrowing
- transition to feature engineering

03_feature_engineering.ipynb
Introduces and evaluates a first conservative feature engineering layer, then compares model routes and recipe variants.

This notebook includes:
- engineered variable creation
- base vs FE route comparison
- cross-validated route evaluation
- recipe selection per model family
- persisted handoff artifacts for the next stage

04_model_refinement.ipynb
Performs finalist refinement and final model selection.

This notebook includes:
- finalist reconstruction
- tuning of final routes
- probability review
- threshold analysis
- fine-grained threshold refinement
- final model selection
- lightweight model interpretation

--------------------------------------------------------------------------------
FEATURE ENGINEERING
--------------------------------------------------------------------------------

A conservative first feature engineering iteration was selected for the project.

Selected engineered features

Core engineered features:
- waist_to_height_ratio
- central_obesity_whtR_flag
- bmi_category
- pulse_pressure
- mean_arterial_pressure
- bp_category_accaha2017
- frequent_snoring_flag
- possible_sdb_risk
- sleep_symptom_burden
- poverty_flag

Additional selected features:
- weekend_sleep_category
- age_band
- education_group

Explicitly excluded for the first iteration:
- sedentary_high_flag
- any rescue of previously dropped variables
- catch-up sleep based on sleep_hours_weekday

Feature engineering rationale is intended to live primarily outside the notebook in dedicated documentation, keeping notebooks clean and technically focused.

--------------------------------------------------------------------------------
MODEL DEVELOPMENT STRATEGY
--------------------------------------------------------------------------------

The project did not treat model selection as a one-shot leaderboard exercise. Instead, it followed a staged refinement process:
1. compare broad candidate routes
2. evaluate the effect of feature engineering
3. select the best recipe per model family
4. tune finalists
5. compare probability behaviour
6. analyze threshold-dependent operating points
7. select the final model under explicit operational criteria

Finalist routes entering refinement:
- HistGradientBoosting (base feature space)
- CatBoost (feature-engineered feature space)

Recipe selection results

Recipe performance was model-dependent rather than universally “native-is-better”.

HistGradientBoosting
Best recipe:
- hgb_legacy_impute_onehot

CatBoost
Best recipe:
- catboost_native_missing_cat_token_nan_min

This reinforced an important project conclusion: the best preprocessing recipe depends on the model family and dataset context.

--------------------------------------------------------------------------------
EVALUATION FRAMEWORK
--------------------------------------------------------------------------------

No single metric was treated as universally sufficient.

The project evaluated performance across three complementary layers:

1. Operational classification metrics
Used when applying explicit thresholds:
- Precision
- Recall
- F1
- Balanced accuracy
- Specificity
- Confusion matrix counts

2. Ranking / discrimination metrics
Used to compare probability behaviour:
- ROC AUC
- Average precision

3. Probability quality
Used to review the quality of predicted probabilities:
- Brier score

Main operational criterion:
- F1

Important complementary control metric:
- Balanced accuracy

This reflected a desire to balance false positives and false negatives rather than optimize only one side of the classification trade-off.

--------------------------------------------------------------------------------
FINAL MODEL SELECTION
--------------------------------------------------------------------------------

After finalist refinement, local threshold sweeps, focused contender comparison, and a final fine-grained threshold refinement, the selected final model was:

catboost_best_by_f1

Final selected configuration:
- Model family: CatBoost
- Feature space: feature-engineered (fe)
- Recipe: catboost_native_missing_cat_token_nan_min
- Selected threshold: 0.43

Final selected hyperparameters:
- learning_rate = 0.0827
- iterations = 294
- depth = 9
- l2_leaf_reg = 6.5655
- random_strength = 2.9497
- bagging_temperature = 1.0991
- border_count = 253

Why this model won:
CatBoost first emerged as the strongest route in probability-oriented evaluation. After a final fine-grained threshold refinement restricted to the two strongest operational contenders, it also achieved the strongest final operational profile on the holdout evaluation split.

This final result was not assumed in advance; it only emerged after:
- probability review
- threshold sweeps
- contender pruning
- micro-refinement of the decision threshold

--------------------------------------------------------------------------------
LIGHTWEIGHT INTERPRETATION
--------------------------------------------------------------------------------

A minimal interpretation step was included using permutation-based feature importance aligned with the selected thresholded operating point.

The final CatBoost model was driven by a mixture of:
- sleep-pattern features
- demographic context
- snoring-related signals
- broader health and behavioural variables

Among the top-ranked inputs, several feature-engineered variables played a meaningful role, which reinforces the value of the FE stage rather than treating it as decorative complexity.

Notably important engineered variables included:
- weekend_sleep_category
- frequent_snoring_flag
- age_band
- possible_sdb_risk
- central_obesity_whtR_flag
- mean_arterial_pressure
- education_group

--------------------------------------------------------------------------------
PERSISTED ARTIFACTS
--------------------------------------------------------------------------------

The project persists structured handoff artifacts between stages.

Examples from the feature engineering handoff include:
- X_train_base.pkl
- X_test_base.pkl
- X_train_fe.pkl
- X_test_fe.pkl
- y_train.pkl
- y_test.pkl
- hgb_recipe_cv_results.csv
- catboost_recipe_cv_results.csv
- recipe_selection_summary.csv
- finalist_route_metadata.json

These artifacts make downstream refinement more reproducible and reduce notebook coupling.

--------------------------------------------------------------------------------
METHODOLOGICAL NOTE
--------------------------------------------------------------------------------

A practical limitation of the current workflow is that the same holdout split was used for:
- probability review
- threshold refinement
- final model selection

As a result, this split should be interpreted as an evaluation holdout rather than as a fully untouched final test set.

This does not invalidate the modeling logic, but it does affect the strictness of the final evaluation design. In a stricter future iteration, the workflow should include:
- a dedicated validation split for threshold selection and final contender comparison
- a separate untouched test split for one-time final evaluation

This is already an identified design lesson for future evolution of the MARS workflow.

--------------------------------------------------------------------------------
WHAT THIS PROJECT DEMONSTRATES
--------------------------------------------------------------------------------

MORPHEUS is meant to demonstrate more than “I can fit a classifier”.

It is a project about:
- disciplined tabular ML workflow design
- reasoning across multiple evaluation layers
- model-family-aware preprocessing choices
- conservative but useful feature engineering
- threshold-aware operational selection
- reproducible notebook handoff design
- honest reflection on methodological trade-offs

--------------------------------------------------------------------------------
FUTURE IMPROVEMENTS
--------------------------------------------------------------------------------

Potential next iterations include:
- stricter train / validation / test separation
- dedicated final packaging artifacts for deployment-style handoff
- optional calibration-focused follow-up
- cleaner artifact management and .gitignore strategy
- further evolution of MARS toward explicit validation-aware split modes

--------------------------------------------------------------------------------
AUTHOR
--------------------------------------------------------------------------------

Andrés Almozara García

If you want to discuss the project, workflow design, or tabular ML portfolio work, feel free to connect through the relevant repository or profile channels.
gid a
