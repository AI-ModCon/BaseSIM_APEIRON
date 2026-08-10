# Agent Skills

The repository ships task-oriented **agent skills** that walk an AI coding agent
through the common Apeiron workflows. Each skill is maintained for both tools:

- **Claude Code** — `.claude/skills/<name>/SKILL.md`
- **Codex** — `.codex/skills/<name>/SKILL.md`

```{important}
Keep the two trees in sync: a change to a workflow should be reflected in both
`.claude/skills/<name>/SKILL.md` and `.codex/skills/<name>/SKILL.md`.
```

## Available skills

| Skill | What it does |
| --- | --- |
| `install-apeiron` | Add Apeiron as a dependency to **another** project (path/git), verify `import apeiron`, pick CPU vs CUDA PyTorch. |
| `explore-examples` | Run a bundled example (MNIST/CIFAR) to see drift detection + CL in action; picks a config and reports the metrics CSV. |
| `custom-experiment` | Scaffold a harness, data utilities, and TOML for **your own** dataset/model, register it in the example factory, smoke-test, and run. |
| `integrate-apeiron` | Add Apeiron's drift detection / CL to an **existing** training loop; inspects your repo and writes the lightest adapter that fits. |
| `choose-detector` | Pick a drift detector (including whether to combine several into an `EnsembleDetector` and which voting rule to use), tune its settings, then emit or patch a validated `[drift_detection]` block. |

## Choosing between them

```{mermaid}
flowchart TD
    A{What do you want to do?} --> B[Try the framework<br/>on shipped data]
    A --> C[Run on my own<br/>data + model]
    A --> D[Keep my own<br/>training loop]
    A --> E[Just configure<br/>drift detection]
    B --> B1[explore-examples]
    C --> C1[custom-experiment]
    D --> D1[install-apeiron<br/>then integrate-apeiron]
    E --> E1[choose-detector]
```

- **`explore-examples`** vs **`custom-experiment`** — the former runs a bundled
  config, the latter scaffolds everything for your dataset and architecture.
- **`custom-experiment`** vs **`integrate-apeiron`** — use `custom-experiment`
  for a self-contained Apeiron run; use `integrate-apeiron` when you already have
  a PyTorch / Lightning / HF Trainer loop and want to bolt drift detection onto it.
- **`install-apeiron`** is only for adding Apeiron to a *separate* project.
  Developing inside this repo is just `poetry install`.
- **`choose-detector`** stops at a validated config block — it does not run an
  experiment.

## Using them

### Claude Code

The skills are exposed as slash commands. Type `/` and the skill name:

```text
/explore-examples
/install-apeiron ../my-project
/choose-detector examples/mnist/mnist.toml
```

You can also just describe the task in plain language ("add apeiron to my
training loop") and the matching skill triggers from its description.

### Codex

The equivalent skills live under `.codex/skills/`. Invoke a skill by name or
describe the task; Codex selects the skill whose description matches the
request. The skills are tool-agnostic in intent — only the file format differs
between the two trees.

## Authoring notes

Skills should defer to these docs rather than restating numbers that can drift.
For example, `choose-detector` names {doc}`drift_detectors` as its authoritative
reference for detector behavior and options, and re-checks
`src/apeiron/drift_detection/load_drift_detector.py` before relying on which
detectors are wired up.
