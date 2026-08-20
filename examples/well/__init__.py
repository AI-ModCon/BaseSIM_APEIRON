"""The Well example: streaming neural PDE surrogate under regime drift.

Reads `The Well <https://polymathic-ai.github.io/the_well>`_ physics-simulation
data (or a schema-identical fixture), commits time windows to a ``WindowStore`` as
next-step regression pairs ordered by a simulation parameter, and runs apeiron's
drift-detection + continual-learning loop over the resulting drifting stream.
"""
