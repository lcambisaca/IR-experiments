# AgentIR: retrieval experiments with our saved 50k image vectors

This project loads the image embeddings we already computed in Colab, encodes
new text with the matching BLIP model, and ranks **50,000 candidate images**.
Use `run_evaluation.py` to compare the original caption, raw VisDial dialogue,
the authors' supplied PlugIR queries, and your own experiments.

**Mac teammates: follow the steps below in order.** You create your own Python
environment and download our existing vectors once. You do not need to encode
images again, download the image archive, rent a GPU, or supply an API key.
Installation uses a few terminal commands; after setup you can edit a Python
file and click **Run** in VS Code.

## 1. Check your Mac and install the prerequisites

Open **Apple menu > About This Mac** and check the chip:

| Your Mac | Setup route |
|---|---|
| Apple Silicon, such as M1/M2/M3/M4 | Use native ARM Python and the steps below, subject to the package compatibility check |
| Intel processor | Use an existing compatible remote Linux environment through VS Code Remote SSH, or the team's Colab workflow |

Our pinned PyTorch version does not have official native Intel Mac wheels.
Do not use the Windows lock file or downgrade PyTorch to get around that.
[PyTorch's Intel macOS support announcement](https://pytorch.org/blog/pytorch2-2/)
explains the support boundary. See [Mac diagnostics and coding-agent brief](#mac-diagnostics-and-coding-agent-brief) for additional checks.

Install these prerequisites on a compatible Apple Silicon Mac:

- [VS Code](https://code.visualstudio.com/) and Microsoft's **Python** extension.
  Install **Python Debugger** if using Run and Debug.
- **Git**, for cloning/updating the repository. VS Code will tell you if Git
  is unavailable; see its [source control setup guide](https://code.visualstudio.com/docs/sourcecontrol/quickstart).
- A native **Python 3.11** interpreter. Python 3.12 is also allowed by the project,
  but the examples below use 3.11. Do not use Python 3.13+ for this environment.

If you already use Homebrew, install Python with:

```bash
brew install python@3.11
```

If you don't have Homebrew, follow its [official installation instructions](https://brew.sh/)
and terminal configuration steps first, or use another native Python installation
method. The [Python 3.11 formula](https://formulae.brew.sh/formula/python@3.11)
documents its supported macOS versions and interpreter location. A machine whose
macOS cannot install the pinned packages should use the remote/Colab route.

In VS Code, open **Terminal > New Terminal**, then check:

```bash
python3.11 --version
python3.11 -c "import platform; print(platform.machine())"
```

Expect Python **3.11.x** and **`arm64`**. If an Apple Silicon Mac reports `x86_64`,
the terminal or Python may be running under Rosetta; switch to native ARM before
creating the environment. Don't replace macOS's system Python.

Windows CPU has been tested. **The Mac instructions have not yet been executed
on a teammate's Mac**; the checks below establish whether their setup works.
The CPU route is intentional: this project has not implemented/tested Apple MPS
acceleration. Local setup needs space for Python packages and BLIP's roughly
1.9 GB model download in addition to the approximately 52 MB image-vector cache.

## 2. Clone the repository and open the project folder

1. Use [lcambisaca/IR-experiments](https://github.com/lcambisaca/IR-experiments).
   Clone URL: `https://github.com/lcambisaca/IR-experiments.git`.
2. In VS Code press **Cmd+Shift+P**, choose **Git: Clone**, and paste that URL.
3. Choose a local development location, then open the cloned folder.
4. Confirm the opened folder contains **`pyproject.toml`, `experiment.json`, and
   `run_evaluation.py`** directly. If they are inside `agentir-vscode`, open that
   subfolder instead. The GitHub repository name doesn't have to be this name.

The original Colab notebook is separate from this Python runner. It is not
executed or needed by the local evaluation code.

## 3. Create your own environment and install the project (once)

Open a new VS Code terminal in the folder containing `pyproject.toml`.
Run these commands **one line at a time**, checking for errors before continuing:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -c "import sys,platform; print(sys.executable); print(platform.machine())"
```

The printed executable should be inside this project's `.venv/bin/`, and its
architecture should be `arm64`. Your environment is local to your Mac; never
copy the Windows `.venv` or commit an environment to GitHub.

Before installing the large packages, check that matching prebuilt wheels exist:

```bash
python -m pip install --dry-run --only-binary=:all: torch==2.11.0 torchvision==0.26.0
```

This checks compatibility without installing those packages. If it fails, use
the troubleshooting table below rather than letting a PyTorch source build run.
If it succeeds, install the project and run the dependency check:

```bash
python -m pip install -e ".[test]"
python -m pip check
```

What these steps do:

| Step/file | Purpose |
|---|---|
| `python3.11 -m venv .venv` | Creates an isolated environment on this computer |
| `source .venv/bin/activate` | Makes this terminal use that environment's Python |
| `pip install -e ".[test]"` | Reads `pyproject.toml`, installs the required libraries and tests, and links our editable source code |
| `pip check` | Checks installed package dependency requirements; expect no broken requirements |
| `requirements-lock-windows-cpu.txt` | Records the tested Windows installation; **do not install this on Mac** |
| `setup_windows.ps1` | Windows-only installation helper; **do not run it on Mac** |

The direct project requirements include PyTorch 2.11.0, torchvision 0.26.0,
Transformers 4.51.3 and the tokenizer/model-download libraries. Keep those pins;
they match our saved-cache setup. Mac installs resolve their own compatible
platform builds and indirect dependencies. This is not a tested Mac lock file.

## 4. Select the environment in VS Code

Press **Cmd+Shift+P > Python: Select Interpreter** and choose:

```text
<your project folder>/.venv/bin/python
```

If it isn't listed, choose **Enter interpreter path** and browse to that file.
This selection controls Python's Run/Debug features. Activating a terminal alone
isn't a substitute for checking the editor's interpreter.

After selecting it, use **Run Python File** from Microsoft's Python extension,
not an unrelated extension's generic **Run Code** command.
VS Code may display output in its integrated terminal; you still launch your
experiments by clicking Run. The [VS Code environment guide](https://code.visualstudio.com/docs/python/environments)
explains interpreter selection.

## Teammates: get the cache once from Drive

GitHub contains the code and small protocol/reference files. **The shared image
vectors are on Google Drive**, and each teammate needs permission to download
them. GitHub repository access does not automatically grant Drive access.

Download **both** existing files without changing their names:

- [50k BLIP vectors: corpus_blip_eval_d85c7c39be4ea84c.pth](https://drive.google.com/file/d/1V8ajL6VqbR2SV-zvz_BbRmo4xPufrRtw/view)
- [Matching metadata: corpus_blip_eval_d85c7c39be4ea84c.json](https://drive.google.com/file/d/1uRFdwNYAOGYbPBzGO2pMB1yt7CfknR3O/view)

In VS Code's Explorer create `data`, then `cache` inside it, and place both files
there. Your folder should look like this:

```text
<project>/
├── pyproject.toml
├── experiment.json
├── run_evaluation.py
├── .venv/                         # Created on your own computer
├── assets/protocol/manifest_50k.json
└── data/cache/
    ├── corpus_blip_eval_d85c7c39be4ea84c.pth
    └── corpus_blip_eval_d85c7c39be4ea84c.json
```

The `.pth` holds the saved image IDs and `(50000, 256)` embedding tensor. It is
51,602,197 bytes. The `.json` records the model revision, encoding versions and
checksums. The ordered image manifest is already supplied with the repository.

The loader reads **local files**, not live Drive URLs. It verifies checksums,
image order, count, IDs, vector shape and normalization. Leo's existing local
copy isn't included when you clone. Do not rename a 5k cache to this filename.
You don't need the 8.22 GB image archive for these text retrieval experiments.

## 5. Test the installation, then run a small evaluation

First open VS Code's **Testing** panel. If asked to configure tests, choose
**pytest** and the **tests** directory. Run all tests. The latest Windows suite
has **18 passing tests**; no BLIP download, GPU or API is needed for these tests.
The tests check code and synthetic examples; the next step also loads the real
downloaded cache and runs the real retriever.

Open `run_evaluation.py` and set:

```python
N_TARGETS = 4
N_STATES = 3
K_VALUES = [1, 10, 50]
DEVICE = "cpu"
OFFLINE = False
```

**Set these explicitly before the first run**: the current file may reflect a
larger local experiment. Four targets still search all **50,000 candidate images**.
Three states mean the initial caption followed by two saved dialogue turns.

Save the file, then click **Run Python File**. Alternatively, use the
**Run experiment evaluation** preset in Run and Debug.

The runner:

1. Loads and validates the existing image-vector cache into CPU memory.
2. Loads the matching `Salesforce/blip-itm-large-coco` model and processor.
   First use may download approximately 1.9 GB; later runs reuse the model cache.
3. Encodes query text, computes cosine similarity to every saved image vector,
   ranks candidates and measures each known target's position.
4. Prints Recall/Hits tables and BRI, then saves the detailed comparisons.

No image encoding or LLM calls happen. This setup uses supplied dialogue rather
than generating new questions. Once the model files are downloaded, you can set
`OFFLINE = True` to require their local reuse.

## 6. Read the results and add experiments

Every run includes these references:

| Method | Input |
|---|---|
| `caption_only` | Original description, unchanged at every state |
| `raw_dialogue` | Original description plus VisDial dialogue available so far |
| `supplied_plugir` | Authors' previously generated PlugIR query for that state |

New methods are registered in the runner's `EXPERIMENTS` dictionary. The provided
template is disabled; enabling its unchanged body should match `raw_dialogue`.
See [EXPERIMENTS.md](EXPERIMENTS.md) before adding a method that returns text,
a custom query vector, or fused agent scores. Known targets are for measurement
after scoring; don't use them to select which agent wins during retrieval.

Each run creates a fresh `runs/experiments/<timestamp>/` folder:

| File | What to inspect |
|---|---|
| `rank_comparison.csv` | Target ranks side by side for every method/state |
| `per_example.csv` | Queries, previous queries, ranks, scores and rank changes |
| `per_round.csv` | Recall/Hits at each state and K |
| `final_metrics.csv` | Final Recall/Hits and overall BRI by method |
| `report.json` | Full results, selected targets, cache identity and software versions |
| `runner_snapshot.py` | The runner used to produce this run |

Higher Recall/Hits and lower BRI are better. Positive rank improvement means
moving closer to first place. BRI summarizes the sequence and is undefined for
one state. Target cosine alone is not the main outcome: the target must rank
well against competing images.

After the small check succeeds:

- `N_TARGETS = 32`, `N_STATES = 11` selects the earlier Colab reference sample.
- `N_TARGETS = None` selects all **2,064 matched targets**.
- `N_STATES = 11` evaluates the caption plus all ten saved dialogue turns.

Increasing target count or states reuses the same image vectors. It increases
text encoding/ranking work, so keep experiments small while developing ideas.

## When you return for another session

Reopen the project, confirm `.venv/bin/python` is selected, edit the runner,
and click Run. You don't recreate the environment or redownload vectors each
session. The first-time installation commands are not an everyday requirement.
If using a new terminal for package commands, activate the environment again
with `source .venv/bin/activate`. Update dependencies only when project changes
require it; source edits are picked up through the editable installation.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `python3.11: command not found` | Install Python 3.11 and follow the installer's PATH/shell steps; reopen the terminal |
| Apple Silicon Python reports `x86_64` | Check for a Rosetta terminal/interpreter; use a native ARM installation |
| No matching Torch/torchvision distribution | Record macOS, Python and architecture; Intel/unsupported macOS needs the remote/Colab route |
| `ModuleNotFoundError: agentir` | Open the folder with `pyproject.toml`, select its `.venv`, and install with `python -m pip install -e ".[test]"` in the activated terminal |
| Missing cache or metadata | Put both exact filenames under `data/cache/`; check for extra download suffixes |
| Cache or manifest checksum error | Download the original matching files again; don't edit metadata to bypass the check |
| CUDA unavailable | Set `DEVICE = "cpu"`; this Mac setup doesn't use NVIDIA CUDA |
| Model files unavailable with offline mode | Set `OFFLINE = False` for the first download, then switch back if desired |
| Unexpectedly long evaluation | Check `N_TARGETS`; use 4 instead of None for a quick check |

Record the Mac chip, macOS/Python versions, failed command and complete error
when asking for help. Don't change pinned libraries or experiment logic just to
make installation succeed. The section below includes additional diagnostics
and instructions you can give your coding agent.

## Mac diagnostics and coding-agent brief

<details>
<summary>Extra checks and instructions to give your coding agent</summary>

If setup fails, record the operating system, interpreter and architecture from
the activated environment:

```bash
uname -m
sw_vers
python -c "import sys,platform; print(sys.executable); print(platform.machine()); print(platform.python_version())"
```

If package installation succeeded, record the actual versions and a Mac-local
dependency snapshot:

```bash
python -c "import torch,torchvision,transformers; print(torch.__version__,torchvision.__version__,transformers.__version__)"
python -m pip freeze > requirements-lock-macos-local.txt
```

This snapshot is not a replacement for the Windows lock. Before sharing it,
remove machine-specific editable-install paths (`-e ...`) and any private package
URLs. Report the chip, macOS/Python versions, dependency check, test results,
cache validation and reference metrics. Distinguish checks that passed from
checks that haven't been run. Don't start a PyTorch source build as routine setup.

**Brief to copy to your coding agent:**

> Help me set up AgentIR on this Mac. Read README.md, pyproject.toml and
> TESTING.md first. Detect the physical chip, interpreter architecture, macOS
> and Python versions. Create an isolated native .venv; don't copy the Windows
> environment or install its Windows wheel builds/lock file. Keep direct package
> pins when supported and diagnose wheel incompatibility before changing them.
> For Intel or unsupported macOS, explain the restriction and use an existing
> compatible remote environment or the team's Colab workflow; don't provision a
> paid machine or silently downgrade PyTorch. Obtain the two authorized Drive
> cache files without re-encoding or downloading image pixels. Run pip check
> and pytest, then set four targets and three states for a small CPU evaluation.
> Select the environment in VS Code and use Run Python File. Once it works,
> check the 32-target/11-state reference described in TESTING.md. Preserve the
> original cache, manifest, reference results and model revision. Don't call
> paid APIs, implement Qwen, or change experiment logic during setup. Report
> exactly which checks passed and record Mac environment versions separately.

</details>

## Windows setup (Leo / Windows teammates)

In this folder's PowerShell terminal run `setup_windows.ps1`, which creates the
local environment and installs the tested Windows CPU package snapshot.
Then select `.venv/Scripts/python.exe` in VS Code, download the same two cache
files if needed, and use the same click-Run workflow above. Mac users should
follow the Mac steps instead. Historical verification is in [TESTING.md](TESTING.md).

## What is shared through GitHub

Share the source, tests, setup instructions, protocol files and reference data.
The existing `.gitignore` excludes `.venv`, local caches, model weights,
generated results, and `.env` files. Do not add these ignored files
to a commit. Drive access is managed separately from repository access.

## Repository map and current scope

```text
pyproject.toml                 # Project installation recipe and dependency pins
experiment.json                # Local cache/protocol/reference file locations
run_evaluation.py              # Edit settings and methods, then click Run
src/agentir/cache.py           # Validate and load saved image vectors
src/agentir/retrieval.py       # Existing BLIP text encoder and cosine search
src/agentir/evaluation.py      # Reusable experiment comparisons and shared metrics
EXPERIMENTS.md                 # Experiment interface and result interpretation
assets/protocol/              # Image order and matched public query data
assets/colab_reference/        # Saved 32-target reference results/config
tests/                        # Synthetic/offline checks
data/cache/                   # Downloaded Drive files; ignored by Git
runs/                         # Generated results; ignored by Git
```

The frozen image vectors use BLIP revision
`19502f1e215844f7e48bd48473f86932486d3441`. Windows verification reproduced the
earlier **32-target/11-state** Colab reference; Mac execution still needs checking.
The supplied PlugIR and raw VisDial methods may use different conversations,
so their difference is not a controlled rewrite-only result. This runner doesn't
automatically compare against Colab and isn't a full reproduction of the paper.
Qwen integration, new question generation, BLIP-2 live answering, and agent
branch exploration are future work, not part of this installation.

## Attribution

Protocol data, dialogue interpretation and feature/metric formulas follow
[PlugIR](https://github.com/Saehyung-Lee/PlugIR) at commit
`ce79c9ff834aabc9de698efb706c6cb9427ecf35`. Its MIT notice is preserved in
`third_party/PlugIR_LICENSE`. Paper:
[Interactive Text-to-Image Retrieval with Large Language Models](https://arxiv.org/abs/2406.03411).
BLIP uses the public [Salesforce checkpoint](https://huggingface.co/Salesforce/blip-itm-large-coco);
consult its model card for the original model background.
