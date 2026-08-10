# Apeiron

**A PyTorch framework for continual learning that detects concept drift in a live
data stream and adapts the model in place.**

Apeiron runs a monitoring loop over a changing data stream, watches an evaluation
metric, hands that metric to a drift detector, and — when the detector fires —
pauses monitoring to run a continual-learning update on the model before
resuming.

```{code-block} bash
:caption: Run a bundled example

poetry run python -m src.main --config examples/mnist/mnist.toml
```

## The loop

1. Evaluate the current model on stream batches.
2. Aggregate the monitored metric at a configured interval.
3. Run a drift detector on the aggregated metric.
4. On drift, pause monitoring and run a continual-learning update loop.
5. Resume monitoring on the updated model until stream limits are reached.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Get started
:link: quickstart
:link-type: doc

Install Apeiron and run your first drift-detection experiment.
:::

:::{grid-item-card} {octicon}`gear` Configuration reference
:link: configurations
:link-type: doc

Every TOML section and key the config parser accepts, with defaults.
:::

:::{grid-item-card} {octicon}`pulse` Drift detectors
:link: drift_detectors
:link-type: doc

Detector classes, their options, and how detector output drives training.
:::

:::{grid-item-card} {octicon}`sync` Continual learning
:link: continuous_learning
:link-type: doc

The CL trainer, the updater modes, and what runs after drift is detected.
:::

::::

## Suggested reading order

1. {doc}`installation` and {doc}`quickstart` — get a run going.
2. {doc}`architecture` — how the pieces fit together at runtime.
3. {doc}`configurations` — the required and optional configuration parameters.
4. {doc}`model_harness` — how your model and stream loaders are exposed to the framework.
5. {doc}`drift_detectors` — how monitoring decisions are made.
6. {doc}`continuous_learning` — what happens after drift is detected.
7. {doc}`tracking` — sending run metrics to Weights & Biases or MLflow.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
quickstart
architecture
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

configurations
model_harness
drift_detectors
choosing_a_detector
continuous_learning
tracking
```

```{toctree}
:maxdepth: 2
:caption: Operations
:hidden:

profiler
deployment
agent_skills
```

```{toctree}
:maxdepth: 2
:caption: API reference
:hidden:

api/index
```
