#!/usr/bin/env python3
"""
RL Environment Task for LLM Training
------------------------------------
This script defines an RL-style programming task for fine-tuning language models.

The task:
    Implement a robust data-cleaning & aggregation function (aggregate_by_group).
The environment:
    - Provides a clear prompt.
    - Tests model-submitted code on randomized data.
    - Grants binary reward: pass/fail.
    - Reports pass rate over multiple trials.

Usage:
    1. pip install anthropic python-dotenv
    2. Create .env file with:
        ANTHROPIC_API_KEY=sk-...
        ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
        NUM_TRIALS=10
        TEMPERATURE=0.0
    3. python main.py
"""

import os, time, random, statistics, traceback, json
from typing import Any, Dict, List
from dotenv import load_dotenv
import anthropic

# ---------- Load .env ----------
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
N = int(os.getenv("NUM_TRIALS", "10"))
TEMP = float(os.getenv("TEMPERATURE", "0.0"))

# ---------- RL Task Prompt ----------
PROMPT = """
Write a Python function named `aggregate_by_group(rows)` that:
  - Takes `rows`: a list of dicts with keys like 'user', 'score', 'age', 'name'.
  - Groups rows by the 'user' key (ignore missing or empty users).
  - For each user, return:
      "count": number of rows
      "mean_score": mean of numeric 'score' values, or None
      "median_age": median of numeric 'age' values, or None
      "top_names": up to 3 most common non-empty 'name' strings,
                   sorted by frequency desc then alphabetically.
  - Must not crash on missing keys or wrong types.
  - Must not import external libraries.
Submit only the function definition.
""".strip()

# ---------- Grader ----------
def _mean(nums): return float(sum(nums)/len(nums)) if nums else None
def _median(nums): return float(statistics.median(nums)) if nums else None
def _top(names):
    freq={}; [freq.setdefault(n,0) or freq.__setitem__(n,freq[n]+1)
              for n in names if isinstance(n,str) and n]
    if not freq: return []
    return [k for k,_ in sorted(freq.items(),key=lambda kv:(-kv[1],kv[0]))[:3]]

def grade(func)->bool:
    """Property-based grading."""
    tests=[
        ([],{}),
        ([{"user":"A","score":10,"age":20,"name":"X"}],
         {"A":{"count":1,"mean_score":10.0,"median_age":20.0,"top_names":["X"]}})
    ]
    # random test generation
    for s in range(5):
        random.seed(s)
        rows=[]
        for _ in range(40):
            u=random.choice(["A","B","C","","Z"])
            r={"user":u}
            if random.random()<.8:r["score"]=random.choice([None,random.uniform(0,100),"x"])
            if random.random()<.8:r["age"]=random.choice([None,random.randint(18,70),"bad"])
            if random.random()<.7:r["name"]=random.choice(["","bob","alice","bob","carol"])
            rows.append(r)
        tests.append((rows,None))
    for rows,exp in tests:
        try: out=func(rows)
        except Exception as e: print("Crash:",e);return False
        if not isinstance(out,dict):return False
        users={r.get("user") for r in rows if isinstance(r.get("user"),str) and r.get("user")}
        for u in users:
            got=out.get(u)
            if not got or not all(k in got for k in("count","mean_score","median_age","top_names")):return False
            rs=[r for r in rows if r.get("user")==u]
            ref={"count":len(rs),
                 "mean_score":_mean([r.get("score")for r in rs if isinstance(r.get("score"),(int,float))]),
                 "median_age":_median([r.get("age")for r in rs if isinstance(r.get("age"),(int,float))]),
                 "top_names":_top([r.get("name")for r in rs if isinstance(r.get("name"),str)])}
            if got["count"]!=ref["count"]:return False
            for k in("mean_score","median_age"):
                a,b=got[k],ref[k]
                if (a is None)!=(b is None):return False
                if a is not None and abs(a-b)>1e-6:return False
            if got["top_names"]!=ref["top_names"]:return False
    return True

# ---------- Extract / Grade Code ----------
def extract_code(resp:str)->str:
    if "```python" in resp:
        try:return resp.split("```python")[1].split("```")[0].strip()
        except:pass
    if "def " in resp:return resp[resp.index("def "):].strip()
    return resp.strip()

def run_code(code:str)->bool:
    ns:Dict[str,Any]={}
    try:exec(code,ns,ns)
    except Exception as e:print("Exec error:",e);return False
    f=ns.get("aggregate_by_group")
    return bool(callable(f) and grade(f))

# ---------- Anthropic Caller ----------
def call_claude(prompt:str)->str:
    import anthropic
    client=anthropic.Anthropic(api_key=API_KEY)
    resp=client.messages.create(
        model=MODEL,temperature=TEMP,max_tokens=1024,
        messages=[{"role":"user","content":prompt}]
    )
    if isinstance(resp.content,list):
        return "".join(c.get("text","") for c in resp.content if "text" in c)
    return str(resp.content)

# ---------- Example Solution (demo if no API key) ----------
EXAMPLE = '''
def aggregate_by_group(rows):
    g={}
    for r in rows:
        u=r.get("user") if isinstance(r,dict) else None
        if not isinstance(u,str) or not u:continue
        g.setdefault(u,[]).append(r)
    out={}
    for u,rs in g.items():
        s=[r.get("score")for r in rs if isinstance(r.get("score"),(int,float))]
        a=[r.get("age")for r in rs if isinstance(r.get("age"),(int,float))]
        n=[r.get("name")for r in rs if isinstance(r.get("name"),str)]
        freq={};[freq.setdefault(x,0)or freq.__setitem__(x,freq[x]+1)for x in n if x]
        top=[k for k,_ in sorted(freq.items(),key=lambda kv:(-kv[1],kv[0]))[:3]]
        out[u]={"count":len(rs),
                "mean_score":float(sum(s)/len(s)) if s else None,
                "median_age":float(statistics.median(a)) if a else None,
                "top_names":top}
    return out
'''

# ---------- RL Environment Main Loop ----------
def main():
    start=time.time();passes=0;results=[]
    if not API_KEY:
        print("No API key found — running local demo:")
        ok=run_code(EXAMPLE)
        print("Local example passed:",ok)
        return
    print(f"Running RL task with model={MODEL}, trials={N}, temp={TEMP}")
    for i in range(N):
        print(f"\n--- Trial {i+1}/{N} ---")
        try:
            resp=call_claude(PROMPT)
            code=extract_code(resp)
            ok=run_code(code)
            print("Pass:",ok)
            passes+=ok
            results.append({"trial":i+1,"ok":ok})
        except Exception as e:
            print("Trial error:",e);traceback.print_exc()
            results.append({"trial":i+1,"ok":False})
    rate=100*passes/N
    print(f"\nSummary: {passes}/{N} passed ({rate:.1f}%), time {time.time()-start:.1f}s")
    with open("results.json","w")as f:json.dump(results,f,indent=2)
    print("Saved results.json")

if __name__=="__main__":main()
