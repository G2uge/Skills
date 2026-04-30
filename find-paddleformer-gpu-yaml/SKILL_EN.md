---
name: find-paddleformer-gpu-yaml
description: |
  Locate PaddleFormers repository based on environment and intelligently search for the best matching GPU YAML configuration file for the target model.
  Support specifying Python environment, repository path, or YAML folder path. Score and select optimal templates based on semantic similarity.
keywords: paddleformer, gpu, yaml, config, find, search, locate, configuration
---

# Find PaddleFormers GPU YAML Configuration

This Skill guides the Agent to complete two core tasks:
1. **Locate the PaddleFormers code repository** (or use specified path directly)
2. **Search for and select the best matching GPU YAML configuration file**

## Input Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_name` | Yes | Target model name, e.g., "Qwen3-VL-30B-A3B-Instruct" |
| `yaml_dir` | No | Directly specify YAML configuration folder path, highest priority |
| `repo_path` | No | Directly specify PaddleFormers repository root directory |
| `python_path` | No | Specify Python interpreter path to locate paddleformers from specific environment |

**Parameter Priority**: `yaml_dir` > `repo_path` > `python_path` > auto-locate

## Execution Flow Overview

```
Input: model_name + optional params (yaml_dir/repo_path/python_path)
  │
  ▼
┌─────────────────────────────────────────┐
│  Step 1: Determine YAML Search Scope     │
│  - If yaml_dir provided: use directly    │
│  - If repo_path provided: scan in repo   │
│  - If python_path provided: locate env   │
│  - Otherwise: auto layered search        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────┐
│  Step 2: Search YAMLs    │
│  - Scan specified range  │
│  - Extract semantic info │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Step 3: Select Best     │
│  - Model feature match   │
│  - Score and rank        │
│  - Output recommendation │
└─────────────────────────┘
           │
           ▼
Output: Best matching GPU YAML path + selection reasoning
```

## Step 1: Determine YAML Search Scope

Based on user-provided parameters, determine search scope in the following priority:

### Priority 1: Directly Specify YAML Folder (`yaml_dir`)

**If user provides `yaml_dir`**:
- **Use this folder directly as search scope**
- Skip all repository location steps
- Verify the folder exists and contains YAML files

**Validation command**:
```bash
ls {yaml_dir}/*.yaml {yaml_dir}/*.yml 2>/dev/null | head -5
```

**Success indicator**: Returns at least one YAML file path

### Priority 2: Directly Specify Repository Path (`repo_path`)

**If user provides `repo_path`**:
- Use this path as PaddleFormers repository root directory
- Perform repository validation
- Scan YAML files within the repository

**Repository validation**:
```bash
ls {repo_path}/examples/configs/ 2>/dev/null || ls {repo_path}/configs/ 2>/dev/null || ls {repo_path}/tests/config/ 2>/dev/null
```

**Required**: `examples/` or `configs/` or `tests/config/` directory

### Priority 3: Specify Python Environment (`python_path`)

**If user provides `python_path`**:
- Use this Python interpreter to import paddleformers
- Get repository path from this environment

**Execute**:
```bash
{python_path} -c "import paddleformers; print(paddleformers.__path__[0])"
```

**Success indicator**: Returns valid path, e.g., `/root/paddlejob/PaddleFormers/paddleformers`

**Processing**:
- Extract parent directory as repository root
- Verify config directories exist

### Priority 4: Auto Layered Search (Default)

If none of the above parameters are provided, execute auto layered search:

#### Layer 1: Python Runtime Environment (Default python)

**Execute**:
```bash
python -c "import paddleformers; print(paddleformers.__path__[0])"
```

**Processing**:
- Extract parent directory as repository root
- Verify config directories exist

#### Layer 2: Common Development Directories

If Layer 1 fails, check these common locations:

| Check Path | Validation Indicator |
|-----------|---------------------|
| `./PaddleFormers` | `examples/` or `configs/` directory exists |
| `../PaddleFormers` | `examples/` or `configs/` directory exists |
| `~/PaddleFormers` | `examples/` or `configs/` directory exists |
| `/workspace/PaddleFormers` | `examples/` or `configs/` directory exists |
| Environment variable `$PADDLEFORMERS_ROOT` | Points to valid directory |

#### Layer 3: Extended Search (Last Resort)

If all above fail, perform limited global search:

```bash
# Limit depth to 3 to avoid long execution time
find /root /workspace /home -maxdepth 3 -type d -name "PaddleFormers" 2>/dev/null | head -5
```

---

## Step 2: Search for Candidate GPU YAML Files

### 2.1 Scan Configuration Directories

Based on the search scope determined in Step 1, execute corresponding scan:

**Scenario A: Directly specified `yaml_dir`**

Find all YAML files in this folder:
```bash
find {yaml_dir} -type f \( -name "*.yaml" -o -name "*.yml" \)
```

**Note**: Do not limit with `grep -i gpu`, let Agent determine if it's GPU config based on content

**Scenario B: Specified repository path (`repo_path` or located from `python_path`)**

Recursively search for GPU-related YAML files in the repository:

**Search scope**:
```
{REPO_ROOT}/
├── examples/configs/**/*gpu*.yaml
├── examples/config/**/*gpu*.yaml
├── configs/**/*.yaml
├── tests/config/**/*.yaml
└── **/sft/**/*.yaml
└── **/pretrain/**/*.yaml
```

**Execute**:
```bash
find {REPO_ROOT} -type f \( -name "*.yaml" -o -name "*.yml" \) | grep -i gpu
```

### 2.2 Parse Model Features

Extract key features from input `model_name`:

| Feature | Extraction Method | Example |
|---------|------------------|---------|
| `family` | Prefix matching | `qwen3`, `llama`, `deepseek` |
| `variant` | Structure indicator | `vl` (vision), `text`, `audio` |
| `size` | Parameter scale | `7B`, `30B`, `A3B` |
| `structure` | Architecture feature | `moe`, `dense` |
| `task_type` | Training stage | `sft`, `pretrain`, `instruct` |

**Example parsing**:
```
Input: "Qwen3-VL-30B-A3B-Instruct"
  │
  ├── family: "qwen3"
  ├── variant: "vl"
  ├── size: "30B"
  ├── structure: ["moe"]      # Inferred from A3B as MoE
  └── task_type: "instruct"
```

### 2.3 Extract YAML File Semantics

For each candidate YAML file, extract the following information:

**From file path**:
- `path_hints.model_family_from_path`: Model identifier in path (e.g., `qwen3vl`)
- `path_hints.task_from_path`: Task identifier in path (e.g., `sft`, `pretrain`)
- `path_hints.device_from_path`: Device identifier (e.g., `gpu`, `xpu`)

**From file content** (read first 50 lines):
- `model_name_or_path`: Model name in configuration
- `stage`: Training stage
- Key parameters: `per_device_train_batch_size`, `gradient_accumulation_steps`, etc.

---

## Step 3: Select Optimal GPU YAML Template

### 3.1 Scoring Dimensions

Agent scores each candidate file comprehensively:

| Dimension | Weight | Scoring Criteria |
|----------|--------|-----------------|
| **Model Family Match** | 40% | Filename or path contains model family (e.g., `qwen3vl`) |
| **Task Type Match** | 25% | `stage` field matches target task_type |
| **Structure Feature Match** | 20% | Supported structure types (e.g., MoE, VL) match |
| **Config Completeness** | 10% | Required fields are present |
| **Device Compatibility** | 5% | Is GPU configuration (preferred) |

### 3.2 Decision Flow

```
For each candidate file:
  1. Calculate dimension match scores
  2. Compute weighted total score
  3. Record scoring reasoning

Sort by total score descending, select top 3:
  - 1st place score >= 80: Recommend as optimal template
  - 1st place score 60-80: Recommend with manual review reminder
  - 1st place score < 60: List Top-3 for user selection, or indicate no match found
```

### 3.3 Output Format

**Successfully found matching configuration**:
```markdown
✅ Best matching GPU YAML configuration found

📋 Match Result:
   File Path: {relative_path}
   Absolute Path: {absolute_path}

🔍 Model Feature Analysis:
   - Family: {family}
   - Structure: {variant}
   - Scale: {size}
   - Task: {task_type}

📊 Candidate Scoring:
   1. {file1}: {score} points
      - Family Match: ✓
      - Task Match: ✓
      - Reasoning: {reasoning}
   
   2. {file2}: {score} points
      ...

🎯 Recommended Config: {file1}
   Reasoning: {comprehensive selection reason}
```

**No matching configuration found**:
```markdown
⚠️ No highly matching GPU YAML configuration found

📋 Search Results:
   Files Scanned: {N}
   Best Match: {file} (score: {score})

💡 Suggestions:
   1. Verify model name spelling is correct
   2. Check PaddleFormers repository path
   3. Manually specify GPU configuration path
   4. Use reference configuration as alternative
```

---

## Execution Examples

### Example 1: Directly Specify YAML Folder (Recommended, Fastest)

**User Input**:
- `model_name = "Qwen3-VL-30B-A3B-Instruct"`
- `yaml_dir = "/data/configs/gpu"`

**Agent Execution**:

1. **Use Specified Folder Directly**:
   ```bash
   ls /data/configs/gpu/*.yaml
   # Found: qwen3vl_sft.yaml, qwen3vl_pretrain.yaml, llama3_8b_sft.yaml...
   ```

2. **Parse and Score** (Skip repository location):
   ```
   Qwen3-VL-30B-A3B-Instruct features: {family: qwen3, variant: vl, size: 30B, task: instruct}

   Candidate Scoring:
   - qwen3vl_sft.yaml: 85 points (family match + similar task)
   - qwen3vl_pretrain.yaml: 60 points (family match + task mismatch)
   - llama3_8b_sft.yaml: 20 points (family mismatch)
   ```

3. **Output Result**: Recommend `qwen3vl_sft.yaml`

### Example 2: Specify Python Environment

**User Input**:
- `model_name = "Qwen3-VL-30B-A3B-Instruct"`
- `python_path = "/root/paddlejob/Gruge/Gruge_env/paddle/bin/python"`

**Agent Execution**:

1. **Use Specified Python to Locate Repository**:
   ```bash
   /root/paddlejob/Gruge/Gruge_env/paddle/bin/python -c "import paddleformers; print(paddleformers.__path__[0])"
   # Output: /root/paddlejob/PaddleFormers/paddleformers
   ```

2. **Search Candidates**:
   ```bash
   find /root/paddlejob/PaddleFormers -name "*.yaml" | grep -i gpu
   ```

3. **Score and Output Result**

### Example 3: Directly Specify Repository Path

**User Input**:
- `model_name = "Qwen3-VL-30B-A3B-Instruct"`
- `repo_path = "/workspace/PaddleFormers"`

**Agent Execution**:

1. **Validate Repository Path**:
   ```bash
   ls /workspace/PaddleFormers/examples/configs/
   # Confirm config directory exists
   ```

2. **Scan in Specified Repository**:
   ```bash
   find /workspace/PaddleFormers -name "*.yaml" | grep -i gpu
   ```

3. **Score and Output Result**

### Example 4: Auto-locate (Default Method)

**User Input**: `model_name = "Qwen3-VL-30B-A3B-Instruct"` (no other params)

**Agent Execution**:

1. **Auto Layered Search**:
   ```bash
   python -c "import paddleformers; print(paddleformers.__path__[0])"
   # Or try other Layers...
   ```

2. **Scan YAML in Located Repository**

3. **Score and Output Result**

### Example 5: Location Failed Handling

**User Input**: `model_name = "Qwen3-VL-30B-A3B-Instruct"` (auto-locate failed)

**Agent Execution**:
```bash
python -c "import paddleformers..."  # Failed (not installed)
ls ./PaddleFormers  # Does not exist
find /root -maxdepth 3 -name "PaddleFormers"  # Not found
```

**Output**:
```markdown
❌ Unable to locate PaddleFormers repository

Methods attempted:
   1. Python environment import - Failed
   2. Common directory check - Failed
   3. Extended search - Failed

💡 Please specify through one of the following:
   1. Provide yaml_dir: Directly specify YAML configuration folder
   2. Provide repo_path: Directly specify PaddleFormers repository path
   3. Provide python_path: Specify Python environment containing paddleformers
```

---

## Notes

1. **Parameter Priority**: `yaml_dir` > `repo_path` > `python_path` > auto-locate. When high-priority parameter is provided, skip low-priority search steps

2. **Recommended: Use `yaml_dir`**: If config folder path is known, directly specifying `yaml_dir` is the fastest and most reliable way, skipping all location steps

3. **`python_path` Use Case**: When system default python doesn't have paddleformers installed, but a specific virtual environment does, use this parameter to specify environment path

4. **Layered Search Order**: When auto-locating, must execute in Layer 1 → Layer 2 → Layer 3 order

5. **Adjustable Scoring Weights**: If certain dimensions are more important in specific scenarios (e.g., task type priority), Agent can adjust weights

6. **Configuration Completeness Check**: Recommended YAML must contain these required fields:
   - `model_name_or_path`
   - `stage`
   - `per_device_train_batch_size`
   - At least one training data path field

7. **Avoid Over-reliance on Filenames**: Should combine file path, content, and configuration fields for comprehensive judgment, not just filename matching

8. **Timeout Control**: Extended search (Layer 3) should limit time and depth to avoid long blocking
