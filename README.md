# AgentIR

Image-retrieval experiments using our saved **50,000-image BLIP cache**. Our main
comparison is **original PlugIR + Qwen** versus **our independent multi-agent
system**. Each generates its own conversation and query history; both use the
same retrieval/answering models, image pool and evaluator. Agent methods need
not use PlugIR's clustering or question-selection logic. The live Qwen baseline
is implemented; its full GPU pilot is pending.

| Guide | Purpose |
|---|---|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Methods, metrics, outputs and adding experiments |
| [PLUGIR_COMPARISON.md](PLUGIR_COMPARISON.md) | What matches original PlugIR and what the wrapper changes |
| [TESTING.md](TESTING.md) | Running checks and verified results |
| [results/README.md](results/README.md) | Historical baseline and paraphrase experiments |

## 1. Install the project

Install Git, VS Code with Microsoft's Python extension, and Python **3.11**
(3.12 is also supported). Clone
[IR-experiments](https://github.com/lcambisaca/IR-experiments) using VS Code's
**Git: Clone**, then open the folder containing `pyproject.toml`.

### Apple Silicon Mac

Use native ARM Python; `python3.11 -c "import platform; print(platform.machine())"`
should print `arm64`. Homebrew users can install it with `brew install python@3.11`.

In VS Code's terminal, run one line at a time:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --dry-run --only-binary=:all: torch==2.11.0 torchvision==0.26.0
python -m pip install -e ".[test]"
python -m pip check
```

Continue past the dry run only if compatible wheels are available. For Intel
Macs or unsupported macOS versions, use a compatible remote environment or the
team's Colab workflow; our Torch pin has no official native Intel Mac build.
See [PyTorch's support announcement](https://pytorch.org/blog/pytorch2-2/).
Local retrieval uses CPU; Apple MPS is not implemented/tested here.

### Windows

In the project folder's PowerShell terminal, run:

```powershell
.\setup_windows.ps1
```

This creates `.venv` and installs the tested Windows CPU dependencies. Each
teammate creates their own environment; Mac/Linux users use `pyproject.toml`
instead of the Windows lock file or `.venv`.

### Select the VS Code interpreter

Use **Python: Select Interpreter** in the Command Palette and choose:

- Mac/Linux: `.venv/bin/python`
- Windows: `.venv/Scripts/python.exe`

Use **Run Python File** from Microsoft's Python extension. Terminal activation
controls that terminal; the editor's interpreter selection controls Run/Debug.

## 2. Download the cache once

Download both files from Drive into `data/cache/`, keeping their names:

- [corpus_blip_eval_d85c7c39be4ea84c.pth](https://drive.google.com/file/d/1V8ajL6VqbR2SV-zvz_BbRmo4xPufrRtw/view)
- [corpus_blip_eval_d85c7c39be4ea84c.json](https://drive.google.com/file/d/1uRFdwNYAOGYbPBzGO2pMB1yt7CfknR3O/view)

The approximately 52 MB `.pth` contains image IDs and a `(50000, 256)` vector
tensor; the `.json` records encoding identity and checksums. The ordered manifest
is already in Git. `experiment.json` points to these **local copies**; Drive
access is separate from GitHub access. The loader validates the pair and manifest.

Saved-query retrieval needs no JPEG archive, image re-encoding or API key.
The first BLIP load downloads its model weights; later runs reuse them.

## 3. Run a small saved-query check

Open [run_evaluation.py](run_evaluation.py), set these explicitly, save and click Run:

```python
N_TARGETS = 4
N_STATES = 3
K_VALUES = [1, 10, 100]
DEVICE = "cpu"
RUN_LIVE_PLUGIR = False
OFFLINE = False
```

This compares `caption_only`, `raw_dialogue` and `supplied_plugir` against all
50,000 images. It generates no new Q&A. Results go to a fresh
`runs/experiments/<timestamp>/`; [EXPERIMENTS.md](EXPERIMENTS.md#outputs-and-metrics)
explains the tables.

- `N_TARGETS=None`: all 2,064 matched targets; changing target count reuses the cache.
- `N_STATES=11`: caption plus ten interactions. Three states mean caption plus two.
- `START_TARGET`: first matched example; `TEXT_BATCH_SIZE`: text-encoding batch size.
- `OFFLINE=True`: require model files already downloaded on this machine.

The file currently defaults to 32 targets, two states, CPU and live generation off.
For later sessions, reopen the project and reuse its environment/cache.

## 4. Run the live Qwen baseline on an NVIDIA GPU

Connect VS Code to the GPU machine using **Remote SSH**, then select that
machine's Python environment. Create a Linux environment using the venv steps
above. Verify `nvidia-smi` and install CUDA-enabled **Torch 2.11.0 / torchvision
0.26.0** compatible with its driver using the
[official instructions](https://pytorch.org/get-started/previous-versions/).
Then install the project there:

```bash
python -m pip install -e ".[live,test]"
python -m pip check
```

Check CUDA from a Python file:

```python
import torch
print(torch.__version__, torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Copy the same cache/metadata pair to the remote machine. For the first pilot,
change the runner settings to:

```python
N_TARGETS = 1
N_STATES = 3
DEVICE = "cuda"
LIVE_DEVICE = "cuda"
RUN_LIVE_PLUGIR = True
OFFLINE = False
```

Only BLIP-2 needs target pixels. Missing selected JPEGs are downloaded into
`data/images/` when `DOWNLOAD_MISSING_TARGETS=True`. Alternatively, set
`IMAGE_ARCHIVE` to a local `plugir_corpus_50k.tar`; only selected JPEGs are restored.
Drive files must be copied to the machine before ordinary paths can read them.

Qwen and BLIP-2 run locally without OpenRouter/Ollama. Initial weights take many
GB of disk space; models alternate between GPU and host RAM. Check both storage
and memory, especially on a 50 GB disk. Full pretrained inference and peak memory
remain unverified; start with the pilot before increasing targets or states.

Completed rounds and model calls are saved under `runs/live_upstream/<identity>/`.
Matching reruns reuse them; increasing only `N_STATES` extends the completed
prefix. Source/model/cache/runtime changes create a new identity. Use one writer
per checkpoint folder. Live dialogue is also exported with the common results.

## Troubleshooting and sharing

| Issue | Check |
|---|---|
| `agentir` cannot be imported | Correct interpreter and editable project installation |
| Mac wheel unavailable | Python/macOS/architecture; use native ARM or the remote route |
| Cache missing or checksum mismatch | Both original Drive files, correct names and manifest; re-download damaged files |
| Offline model missing | Allow the initial download with `OFFLINE=False` |
| CUDA unavailable | Remote interpreter, NVIDIA driver and CUDA-enabled Torch build |
| Run takes too long | Target/state counts; all searches still include 50k images |

Keep dependency pins from `pyproject.toml`. When requesting setup help, include
OS/chip, Python/package versions, interpreter path and the complete error.

**Brief for a teammate's coding agent:** Read this README, `pyproject.toml` and
`TESTING.md`. Create a native local environment, preserve dependency/cache/model
identities, run the checks, and launch the small saved-query example through VS
Code. Diagnose incompatible wheels before changing pins. Report what passed.

Git shares source, tests, protocol data and curated results. `.venv`, downloaded
models/cache, secrets and new `runs/` outputs remain ignored. Share large files
through Drive with the appropriate access.

## Attribution

[PlugIR paper](https://arxiv.org/abs/2406.03411) and
[source repository](https://github.com/Saehyung-Lee/PlugIR). Pinned original source,
checksums and MIT notice are in [third_party/PlugIR](third_party/PlugIR).
The candidate-caption catalogue is copied from that pinned protocol.
