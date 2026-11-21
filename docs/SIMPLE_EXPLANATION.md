# A simple explanation of your student dropout MLOps project

## 1. What I built

I built the machine-learning part of your project and the process for running and maintaining it.

The model answers this question:

> Using the information available at the end of week 5, how likely is this student to withdraw from this course before it finishes?

For now, all students and outcomes are fictional. This lets you demonstrate the MLOps process without needing university records.

Your colleague still builds the frontend, backend and Moodle connection. Your component receives data and returns predictions.

The assignment's main achievement is a model that can be trained, checked, recorded, packaged, used and monitored in a repeatable way.

## 2. The whole system in one picture

There are two different activities.

**Training means creating a model from historical examples.**

```text
Historical LMS data + known withdrawal outcomes
                    |
                    v
             Check the data
                    |
                    v
             Train the model
                    |
                    v
           Measure its performance
                    |
                    v
          Check the release requirements
                    |
                    v
          Save a passing model version
                    |
                    v
          Explicitly approve it for use
```

**Scoring means using a saved model to make new predictions.**

```text
Your colleague exports LMS activity
                    |
                    v
        Your component checks the file
                    |
                    v
        The saved model predicts probabilities
                    |
                    v
         A results file goes to the backend
                    |
                    v
       The frontend displays the results
```

Opening a dashboard does not train a model. Making predictions does not retrain it either. These are separate operations.

## 3. The data I prepared

I created 3,600 fictional student-course records covering six teaching periods.

Each record is a snapshot of one student's position in one course at day 35. It includes information such as:

| Information | Simple meaning |
|---|---|
| Active days | On how many days did the student use the LMS? |
| Recent interactions | How much activity happened in the last two weeks? |
| Previous interactions | How much activity happened in the two weeks before that? |
| Last activity | How long has it been since the student last used the LMS? |
| Missed assignments | How many assignments were due but had no submission? |
| Late submissions | How many submitted assignments arrived late? |
| Available early grades | What marks were already available by week 5? |

Names and email addresses are unnecessary. Pseudonymous student keys identify rows so the backend can match predictions to its own records.

I separated the data into two files:

- [lms_snapshots.csv](../data/raw/lms_snapshots.csv) contains what was known at week 5.
- [student_outcomes.csv](../data/raw/student_outcomes.csv) contains what eventually happened.

That separation matters. A prediction must use information available at the time. It must not accidentally use a later withdrawal date or final result.

For this target, withdrawal is the positive outcome. A student who fails but stays enrolled is not counted as a withdrawal. Unknown outcomes are not silently treated as students who stayed.

I divided the records into:

| Group | Records | Purpose |
|---|---:|---|
| Training | 2,400 | Teach the model using earlier cohorts |
| Validation | 600 | Check the model and choose an alert threshold |
| Final test | 600 | Assess the frozen model once on a later cohort |

The final test has already been completed. It must not become a dataset for repeatedly adjusting the model.

## 4. The model itself

I used **logistic regression**, a relatively simple model that estimates a probability from numerical inputs.

The complete saved model also includes data preparation. For example, an unavailable early grade stays missing in the export, and the saved preparation step handles that missing value consistently.

The model receives 14 selected numerical features. Student identifiers, outcome columns and other metadata are excluded from its inputs.

An output of `0.24` means an estimated **24% withdrawal probability** under this synthetic demonstration. It is an estimate, not a statement that the student will withdraw.

The model also returns a simple display label:

- Below the saved threshold: `routine`.
- At or above the saved threshold: `review`.

The selected demonstration threshold is `0.15`, meaning 15%. This is an assignment setting, not an approved university policy. The threshold is saved with the model so it cannot silently change between training and use.

## 5. What the four main tools do

### DVC keeps a record of the dataset

Think of DVC as a way to remember which exact version of the data was used.

If the dataset changes later, you need to distinguish the old and new versions. Otherwise, it is difficult to explain why two training runs produced different results.

I tracked the dataset with DVC, copied it to a local recovery store, restored it in a disposable checkout and compared its checksum. A checksum is a fingerprint of a file's contents. The restored file matched exactly.

The recovery store is on the same computer. It demonstrates recovery mechanics; it is not an off-device backup.

### Metaflow runs the training steps in order

Think of Metaflow as the manager of a recipe.

Our recipe has eight steps:

| Step shown in the viewer | What happens |
|---|---|
| `start` | Read the settings and create a tracking record |
| `validate` | Check the export and select usable training and validation records |
| `prepare` | Prepare the information needed by the model |
| `train` | Fit the model on the training records |
| `evaluate` | Measure its performance on the validation records |
| `gate` | Decide whether it meets the release requirements |
| `register` | Register a candidate only if the gate passes |
| `end` | Finish and record the outcome |

If data validation fails, the workflow stops before training. Metaflow records the step history and creates browser cards for inspection.

### MLflow keeps the experiment notebook and model catalogue

Think of MLflow as a notebook that automatically records each experiment, plus a catalogue of saved model versions.

For each training run, it records settings, performance numbers, data fingerprints, reports and the saved model.

This allows you to answer questions such as:

- Which data and settings produced this model?
- What were its measured results?
- Did it pass the release gate?
- Which saved version are predictions using?

The currently approved model has the name `champion`. In the final demonstration registry, that is version 1.

Creating a candidate does not automatically make it the champion. Approval is a separate action.

### Docker packages the environment

Think of Docker as a prepared box containing Python, the required libraries and your project code.

Your colleague can run the packaged scoring job without manually recreating all the library versions.

The same project image supports the MLflow server, training jobs and scoring jobs. Starting the MLflow server does not automatically start training.

I tested predictions both on the host and inside Docker. Their probabilities matched to numerical precision. I also restarted MLflow and confirmed that its saved models and experiment history remained available.

The tested image targets ARM64, matching this computer. It has not been claimed as a native AMD64 build.

## 6. A real output example from the fictional data

The saved [example output](../outputs/phase6_smoke_v1.csv) contains these rows. Percentages below are rounded for readability.

| Fictional student | Course | Predicted withdrawal probability | Display label | Model version |
|---|---|---:|---|---:|
| SYN_003589 | COURSE_01 | 7.8% | routine | 1 |
| SYN_003590 | COURSE_02 | 8.8% | routine | 1 |
| SYN_003591 | COURSE_03 | 24.3% | review | 1 |
| SYN_003592 | COURSE_04 | 29.2% | review | 1 |

The first two probabilities are below 15%. The next two are above 15%, so they receive the `review` label.

The output also records the prediction time, data fingerprint and batch identifier. These help explain where a displayed result came from.

The frontend should present these as estimates. No automatic student message or intervention follows a prediction.

## 7. The release gate prevents automatic replacement

A release gate is simply a set of checks a candidate must pass before it can be approved.

For this demonstration, those checks include finding enough simulated withdrawals, avoiding too many false alerts, comparing probability error with a simple baseline, and checking data lineage. A replacement candidate is also compared with the current champion on the same validation records.

I tested the gate using a deliberately weak constant predictor. It gave every student the same probability based on the training withdrawal rate.

It failed the existing requirements. The gate recorded the reasons, no model version was registered for that baseline, and the approved model stayed unchanged.

This is useful MLOps evidence: the system can reject a candidate even when the computation itself finishes successfully.

## 8. Rollback lets you return to a previous version

Rollback means restoring a previously approved model and its saved settings.

In the earlier host-local registry demonstration, I promoted a second passing version and then restored version 1. The restored information included the model, threshold and schema version.

The final Docker-backed registry is a separate demonstration store. Its approved and assessed model is version 1. The release history labels these two contexts; they should not be presented as one uninterrupted registry history.

## 9. Monitoring checks whether incoming data changes

A model can continue running successfully even when the incoming data becomes different from its training data.

For example, a new cohort might have much less LMS activity. That could reflect student behaviour, a course-design change or a broken export. The alert alone cannot tell which explanation is correct.

I saved a reference profile from training records only. Monitoring compares new batches with that profile for active days, interaction counts and days since last activity.

I demonstrated two cases:

| Input batch | Result |
|---|---|
| The unchanged training reference | No drift detected; all three comparison scores were zero |
| A prepared batch with sharply reduced activity | Drift alert on all three monitored features |

The comparison score is called PSI. For the demonstration, a value above 0.20 raises an alert. Batches with fewer than 100 valid records are labelled too small for this assessment.

The altered batch had no new confirmed outcomes. Its retraining state therefore remained `data_not_ready`.

The important distinction is that **a drift alert is a reason to investigate, not permission to retrain blindly or replace the model**. New supervised training requires confirmed outcomes, data checks, another evaluation and explicit approval.

## 10. What the browser views show

### MLflow

Open [the saved version-1 training record](http://127.0.0.1:5000/#/experiments/1/runs/980c428a6c444c6f9f2e84fb6a696c00).

On its Overview page:

1. **Metrics** are the measured performance numbers. Recall describes how many eventual withdrawals were found. Precision describes how many flagged students actually withdrew.
2. **Parameters** are the settings, library versions and data fingerprints used for that run.
3. **Artifacts** are saved files such as evaluation reports, plots and the deployment bundle.
4. **Registered models** links to the saved version in the model catalogue.

The main [MLflow page](http://127.0.0.1:5000) also contains separate experiments for the failed-gate demonstration and final assessment.

### Metaflow

Open [the Local Card Viewer](http://127.0.0.1:8324).

1. The dropdown lists the eight steps.
2. Select a step to inspect its card.
3. `Task Is Complete` means that particular step finished.
4. The diagram labelled `DAG` shows the order of the steps. DAG is the technical name for this workflow diagram.

The card can show code and intermediate information. You do not need to understand every code line to explain the process.

MLflow answers, “What model and results did this experiment produce?” Metaflow answers, “Which steps ran, and what happened in each step?”

Both views are served by this computer. Their links work while the corresponding services are running. Startup instructions are in [the integration guide](INTEGRATION.md).

## 11. What your colleague needs to connect

Your colleague's backend should:

1. Read the LMS and produce a file matching [the export specification](LMS_EXPORT_SPEC.md).
2. Invoke the Docker scoring job with a specific saved model version.
3. Read the returned results file.
4. Store the results and display them in the frontend under the correct course permissions.

Historical outcome records are needed for training and later evaluation. They are not required when scoring an unlabelled batch.

There is currently no prediction API. The agreed interface is a batch input CSV and a batch output CSV. The [integration guide](INTEGRATION.md) provides the exact commands.

## 12. What has actually been checked

- **45 automated tests passed.** They cover data rules, feature preparation, model/gate behaviour, scoring and monitoring.
- A complete training workflow ran locally and in Docker.
- DVC data recovery matched the original checksum.
- A poor baseline failed the release gate without replacing the champion.
- Promotion and rollback were demonstrated in the earlier local registry.
- Docker scored a complete 12-row batch before and after restarting MLflow.
- Host and Docker probabilities matched to numerical precision.
- Clean and changed-data monitoring cases produced the expected different outcomes.
- The final held-out cohort was assessed once at the frozen threshold.
- Git contains a local source checkpoint. Earlier bootstrap training runs predate that checkpoint and did not record a Git commit.

I also created a GitHub Actions workflow. That file describes automated checks and an image build, but no hosted GitHub run has been verified because no remote repository was connected.

## 13. How good is the model?

The final fictional cohort contained 600 records, including 124 withdrawals.

At the frozen 15% alert threshold, recall was about 83% and precision was about 29%.

In simple terms, it found roughly 83 out of every 100 simulated withdrawals, but only about 29 out of every 100 flagged records were actual simulated withdrawals. This shows the cost of false alerts.

Final precision was slightly below the illustrative 30% validation requirement. I recorded that result and did not adjust the model after seeing the final test.

These numbers describe the generated simulation. They do not show how well the model would work on real university students. Real deployment would require approved historical data, evaluation on that population and an agreed support policy.

## 14. A short explanation you can give your professor

> I implemented a small MLOps lifecycle for a student course-withdrawal predictor. I used a simple logistic-regression model so the focus stays on the operating process. DVC records the dataset version, Metaflow runs the checked training steps, MLflow records experiments and model versions, and Docker packages the environment. The demonstration shows invalid-data rejection, a failed release gate, explicit approval and rollback, consistent batch predictions and drift monitoring. All records are synthetic, so the results demonstrate the lifecycle rather than real-world student prediction quality.

For the step-by-step demonstration and saved evidence, use [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md).
