# 🧠 [AutoGraderRL] RL Environment Task – Data Aggregation Challenge

This project defines a **Reinforcement Learning (RL) task** for training or evaluating large language models (LLMs).  
The goal is for the model to **learn to write a robust data-cleaning and aggregation function**—a practical and valuable skill for machine learning engineers and researchers.

---

## 🎯 Objective

Implement a Python function named `aggregate_by_group(rows)` that:
- Groups a list of dictionaries by `'user'`.
- Computes for each user:
  - `count`: number of rows
  - `mean_score`: average of numeric `'score'` values
  - `median_age`: median of numeric `'age'` values
  - `top_names`: up to three most frequent `'name'` values  
- Must handle missing keys, `None`, and bad data types gracefully.  
- Must not import external libraries.

The environment checks submissions through **property-based grading** with randomized tests.

---

## 🧩 Files

| File | Description |
|------|--------------|
| `main.py` | The entire RL environment (prompt, grader, and harness). |
| `.env` | Configuration file for model and API key. |
| `results.json` | Auto-saved results (pass/fail per trial). |

---

## ⚙️ Setup

1. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```
2. **Activate the Virtual Environment:**
   ```
   venv\Scripts\activate
   ```   
3. **Clone the repository:**
   ```
   git clone
   ```
4. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
5. **Navigate to the project directory:**
   ```
   cd
   ```
6. **Create a .env file:**
   ```
   ANTHROPIC_API_KEY=sk-your_api_key_here
   ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
   NUM_TRIALS=10
   TEMPERATURE=0.0
   ```
7. **Run the agent:**
   ```
   python main.py
   ```

## 🧪 RL Learning Signal

* Reward: 1 for a correct implementation (passes all tests), otherwise 0.
* State: model receives the natural language task description.
* Action: model generates Python code.
* Environment feedback: the grader evaluates code and provides binary reward.

This setup allows Reinforcement Learning from Environment Feedback (RLEF), analogous to RLHF but task-specific.

## 📊 Example Output
```
Running RL task with model=claude-3-5-sonnet-20240620, trials=10, temp=0.0

--- Trial 1/10 ---
Pass: True

Summary: 4/10 passed (40.0%), time 92.4s
Saved results.json
```

