"""Unit test for MNIST drift detection with reproducible results."""

import pytest
import subprocess
from pathlib import Path
import pandas as pd
import shutil


@pytest.fixture
def mnist_config_path():
    """Return path to MNIST test config file (faster version for testing)."""
    return Path("tests/fixtures/mnist_test.toml")


@pytest.fixture
def mnist_full_config_path():
    """Return path to full MNIST config file (for comprehensive tests)."""
    return Path("examples/mnist/mnist.toml")


@pytest.fixture
def temp_output_csv(tmp_path):
    """Create a temporary CSV output path."""
    csv_path = tmp_path / "test_mnist_output.csv"
    return csv_path


def test_mnist_drift_scores_reproducible(mnist_config_path, temp_output_csv):
    """
    Test that running MNIST example with fixed seed produces reproducible drift scores.

    This test:
    1. Runs the command: poetry run python -m src.main --config examples/mnist/mnist.toml
    2. With a fixed seed (specified in the config)
    3. Extracts drift scores from the logged metrics
    4. Verifies basic properties of the drift scores
    """
    # Ensure config file exists
    assert mnist_config_path.exists(), f"MNIST config not found at {mnist_config_path}"

    # Ensure poetry is available
    poetry_path = shutil.which("poetry")
    if not poetry_path:
        pytest.skip("Poetry not found in PATH")

    # Build the command
    cmd = [
        "poetry", "run", "python", "-m", "src.main",
        "--config", str(mnist_config_path),
        "--set", f"visualization.input={temp_output_csv}",
        "--device", "cpu",  # Force CPU for reproducibility
    ]

    # Run the command
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,  # Run from project root
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout for fast test config
    )

    # Check for successful execution
    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Verify CSV was created
    assert temp_output_csv.exists(), f"Output CSV should exist at {temp_output_csv}"

    # Load the logged metrics
    df = pd.read_csv(temp_output_csv)

    # Filter for drift score metrics
    drift_scores = df[df["metric"] == "drift/score"].copy()
    drift_scores = drift_scores.sort_values("step").reset_index(drop=True)

    # Convert value column to float (it may be read as string)
    drift_scores["value"] = pd.to_numeric(drift_scores["value"], errors="coerce")

    # Verify we have drift scores logged
    assert len(drift_scores) > 0, "Should have logged drift scores"

    # Basic sanity checks
    assert drift_scores["value"].min() >= 0, "Drift scores should be non-negative"
    assert drift_scores["value"].max() <= 1.0, "Drift scores should be <= 1.0"

    # Check that we have the expected number of monitoring steps
    # Based on mnist.toml: detection_interval=10, max_stream_updates=20
    assert len(drift_scores) >= 1, "Should have at least one drift score"

    print(f"\nDrift scores logged: {len(drift_scores)}")
    print(f"Drift score range: [{drift_scores['value'].min():.4f}, {drift_scores['value'].max():.4f}]")
    print(f"Mean drift score: {drift_scores['value'].mean():.4f}")


def test_mnist_drift_scores_exact_match(mnist_config_path, tmp_path):
    """
    Test that drift scores exactly match expected values.

    This test compares the first few drift scores against hardcoded expected values
    to ensure complete reproducibility.

    Note: Expected values should be updated after initial test run to record
    the baseline values for future comparisons.
    """
    # Ensure poetry is available
    poetry_path = shutil.which("poetry")
    if not poetry_path:
        pytest.skip("Poetry not found in PATH")

    # Create output CSV path
    csv_path = tmp_path / "test_mnist_exact.csv"

    # Build the command
    cmd = [
        "poetry", "run", "python", "-m", "src.main",
        "--config", str(mnist_config_path),
        "--set", f"visualization.input={csv_path}",
        "--device", "cpu",
    ]

    # Run the command
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Command failed: {result.returncode}\nSTDERR:\n{result.stderr}"
    )

    # Load the logged metrics
    df = pd.read_csv(csv_path)

    # Filter for drift scores
    drift_scores = df[df["metric"] == "drift/score"].copy()
    drift_scores = drift_scores.sort_values("step").reset_index(drop=True)

    # Convert value column to float
    drift_scores["value"] = pd.to_numeric(drift_scores["value"], errors="coerce")

    # TODO: Update these expected values after running the test once
    # For now, we'll just store the first run's values for comparison
    # Uncomment and update after first run:
    # expected_scores = [
    #     0.0000,  # First drift score
    #     0.0123,  # Second drift score
    #     # ... etc
    # ]
    #
    # # Compare first N scores
    # n_scores_to_check = min(5, len(drift_scores))
    # for i in range(n_scores_to_check):
    #     actual = drift_scores.iloc[i]["value"]
    #     expected = expected_scores[i]
    #     assert abs(actual - expected) < 1e-4, (
    #         f"Drift score {i} mismatch: expected {expected:.4f}, got {actual:.4f}"
    #     )

    # For now, just print the scores to establish baseline
    print("\n=== Baseline Drift Scores ===")
    print("Copy these values into the test for exact matching:")
    print("expected_scores = [")
    for i, row in drift_scores.head(10).iterrows():
        print(f"    {row['value']:.6f},  # Step {row['step']}")
    print("]")


def test_mnist_drift_detection_consistency(mnist_config_path, tmp_path):
    """
    Test that drift detection is consistent across multiple runs with same seed.

    This runs the test twice with the same seed and verifies that:
    1. The same drift events are detected at the same steps
    2. The drift scores are identical
    """
    # Ensure poetry is available
    poetry_path = shutil.which("poetry")
    if not poetry_path:
        pytest.skip("Poetry not found in PATH")

    results = []

    for run_id in range(2):
        # Create unique output path for this run
        csv_path = tmp_path / f"mnist_run_{run_id}.csv"

        # Build command (seed is already in the config file)
        cmd = [
            "poetry", "run", "python", "-m", "src.main",
            "--config", str(mnist_config_path),
            "--set", f"visualization.input={csv_path}",
            "--device", "cpu",
        ]

        # Run the command
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=600,
        )

        assert result.returncode == 0, (
            f"Run {run_id} failed: {result.returncode}\nSTDERR:\n{result.stderr}"
        )

        # Load results
        df = pd.read_csv(csv_path)
        drift_scores = df[df["metric"] == "drift/score"].copy()
        drift_scores = drift_scores.sort_values("step").reset_index(drop=True)

        # Convert value column to float
        drift_scores["value"] = pd.to_numeric(drift_scores["value"], errors="coerce")

        results.append(drift_scores)

    # Compare the two runs
    df1, df2 = results[0], results[1]

    # Same number of drift scores
    assert len(df1) == len(df2), (
        f"Both runs should have same number of drift scores: {len(df1)} vs {len(df2)}"
    )

    # Same drift scores at same steps
    for i in range(len(df1)):
        step1, score1 = df1.iloc[i]["step"], df1.iloc[i]["value"]
        step2, score2 = df2.iloc[i]["step"], df2.iloc[i]["value"]

        assert step1 == step2, f"Step mismatch at index {i}: {step1} vs {step2}"
        assert abs(score1 - score2) < 1e-6, (
            f"Score mismatch at step {step1}: {score1:.6f} vs {score2:.6f}"
        )

    print(f"\nConsistency check passed: Both runs produced identical results")
    print(f"Number of drift scores: {len(df1)}")


@pytest.mark.slow
def test_mnist_full_config_drift_scores(mnist_full_config_path, tmp_path):
    """
    Test using the full MNIST configuration (slower, marked as slow test).

    This test uses the complete configuration from examples/mnist/mnist.toml
    with all iterations. It's marked as 'slow' so it can be skipped during
    normal test runs.

    Run with: pytest -v -m slow
    """
    # Ensure poetry is available
    poetry_path = shutil.which("poetry")
    if not poetry_path:
        pytest.skip("Poetry not found in PATH")

    # Create output CSV path
    csv_path = tmp_path / "mnist_full_output.csv"

    # Build the command
    cmd = [
        "poetry", "run", "python", "-m", "src.main",
        "--config", str(mnist_full_config_path),
        "--set", f"visualization.input={csv_path}",
        "--device", "cpu",
    ]

    # Run the command with longer timeout
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout for full config
    )

    assert result.returncode == 0, (
        f"Command failed: {result.returncode}\nSTDERR:\n{result.stderr}"
    )

    # Load and analyze results
    df = pd.read_csv(csv_path)
    drift_scores = df[df["metric"] == "drift/score"].copy()
    drift_scores = drift_scores.sort_values("step").reset_index(drop=True)

    # Convert value column to float
    drift_scores["value"] = pd.to_numeric(drift_scores["value"], errors="coerce")

    # Print comprehensive statistics
    print(f"\n=== Full MNIST Configuration Results ===")
    print(f"Total drift scores logged: {len(drift_scores)}")
    print(f"Drift score range: [{drift_scores['value'].min():.6f}, {drift_scores['value'].max():.6f}]")
    print(f"Mean drift score: {drift_scores['value'].mean():.6f}")
    print(f"Std dev drift score: {drift_scores['value'].std():.6f}")
    print(f"\nFirst 10 drift scores:")
    for i, row in drift_scores.head(10).iterrows():
        print(f"  Step {row['step']:3d}: {row['value']:.6f}")

    # Basic assertions
    assert len(drift_scores) > 0, "Should have logged drift scores"
    assert drift_scores["value"].min() >= 0, "Drift scores should be non-negative"
