# Running the multi-agent pilot in the merged runner

This handoff is for the teammate's `caption-mix-fact-fusion` branch.
Keep that branch's runner, live fact-fusion code, and existing results.
Copy the multi-agent package files from this checkout:

- `src/agentir/multagent_experiment/__init__.py`
- `src/agentir/multagent_experiment/setup.py`
- `src/agentir/multagent_experiment/multi_agent.py`
- `src/agentir/multagent_experiment/agent_prompts.py`

Port the `max_output_tokens` constructor argument and validation from this
checkout's `src/agentir/live_models.py` to the merged `Qwen3` transport.
Use that instance limit in the request-time check as well: the default stays
512 for existing PlugIR calls, while the agents explicitly request a 1024
ceiling for batch filtering. Preserve the rest of the teammate's transport.

In the merged runner, add these imports and register the method alongside
any existing prepared experiments:

```python
from agentir.multagent_experiment.setup import (
    N_STATES as PILOT_STATES, prepare_three_agents, select_pilot_pairs,
)

PREPARED_EXPERIMENTS["three_agents"] = prepare_three_agents
N_STATES = PILOT_STATES
DEVICE = "cuda"
RUN_LIVE_PLUGIR = True
SEARCH = "facts"
```

Place the assignments after their existing settings and registry definitions.
For this pilot, replace the contiguous `START_TARGET` / `N_TARGETS` slicing
in `main()` with `pairs = select_pilot_pairs(pairs)`, immediately after
recording `available = len(pairs)`. This gives every method cake and giraffe
in the same order and avoids the default 32-target selection. If keeping the
original defaults in the same runner, guard both the registration/settings and
the target-selection change with a single pilot flag.

Run the merged entry point on the GPU. Its existing evaluator exports both live
methods to the same rank CSV and keeps the saved references. This compares
complete systems with independent Q&A trajectories. The caption-mix post-hoc
script is separate, and the pilot does not enable critique or run 30 cases.
