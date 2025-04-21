# Agent‑Framework Manual

_A practical guide for developers who want to build **modular, message‑passing pipelines** with Python._

---

## 1  Why this framework exists

Large language‑model (LLM) and data‑processing pipelines often consist of **independent “agents”** that:  
* receive structured input,  
* transform or enrich it,  
* emit new structured output.

You frequently want to **chain** such agents, **fan‑out** one agent’s output to several others, or **aggregate** results that were produced in parallel.  Logging, error‑recovery, and the ability to _pause ↔ resume_ long‑running jobs are non‑negotiable.

This framework gives you:

* **Message ports** (input/output) with optional transformation and conditional routing.
* A **round‑robin scheduler** that treats every agent fairly and can be snapshotted to disk.
* **Parent‑tracking IDs** so that outputs can be reliably re‑assembled downstream.
* A handful of pre‑built agents (debug, save/load JSON, counter etc.) that illustrate common patterns.

---

## 2  Big‑picture architecture

```text
┌────────────┐     ToolPort.send()       ┌────────────┐
│ Connected  │ ───────────────────────▶ │ Connected  │
│  Agent A   │                          │  Agent B   │
└────────────┘ ◀─────────────────────── │            │
           ToolPort.receive()           └────────────┘
```

1. **Agents** inherit from `ConnectedAgent`. Each has
   * exactly one **output port** (built in), and
   * one or many **input ports** (one by default; multi‑port agents override this).
2. **Ports** are wired together at build‑time.  A port can optionally:
   * transform the message (`transformer: msg → msg′ | [msg′…]`),
   * drop messages that don’t satisfy a `condition(msg) → bool`.
3. The **AgentScheduler** iterates round‑robin over all registered agents.  Every call to `agent.step()` consumes **at most one message** from that agent’s input queue.
4. **Parents chain**: each time a message is passed across a port, it gets a synthetic parent ID `UUID:index:len`.  Those IDs let downstream agents figure out which partials belong together.

---

## 3  Core classes – detailed walkthrough

### 3.1  `ToolPort`
| Responsibility | Notes |
| -------------- | ----- |
| FIFO **queue** for incoming messages (input ports) | `queue: deque[(parents, msg)]` |
| Hold **connections** (output → input) | Each connection stores `(target_port, transformer, condition, (source_agent, target_agent))` |
| `send(msg, parents)` | *If* connections exist: broadcasts the message (with new parent ID) to each target. *Else* drops it into `unconnected_outputs` so you can inspect sinks later. |
| `receive()` | Enqueues message into an input port. |

> **ID format**: `msg_uuid:index:total`. Example `33aa…:2:5` means “third chunk out of five”.

### 3.2  `ConnectedAgent`

| Member | Purpose |
| --- | --- |
| `input_schema` / `output_schema` | Pydantic models that enforce structure (may be `NullSchema`). |
| `_input_port`, `_output_port` | Created on construction unless you are a multi‑port agent. |
| `feed(msg)` | Manually injects a message (useful for seeding pipelines). |
| `step()` | **Public – called by the scheduler.**<br>• Pops exactly one item from `_input_port.queue`.<br>• Delegates to `process()` → `run()`. |
| `process(params, parents)` | Default just calls `run(params)` – override when you need `parents`.
| Persistence | `save_state()` / `load_state()` serialise the agent’s pydantic state **plus every port** (queues, dangling outputs). |

### 3.3  `AgentScheduler` (+ `AgentSchedulerState`)

| Feature | Mechanics |
| ------- | --------- |
| **Round‑robin stepping** | `step_all()` loops until every agent reported "nothing to do" for a full pass. |
| **Persistence** | `save_agents(dir)` dumps every agent; `save_state(dir)` dumps a JSON with `agent_idx`, `step_counter`, `all_done_counter`. |
| **Error capture** | If `agent.step()` raises, it is wrapped in `SchedulerException` and optionally persisted to `error_dir`. |
| **Progressive checkpoints** | When `save_dir` is set, the scheduler snapshots every `save_step` rounds. |


---

## 4  Batteries‑included agents

| Agent | What it does | Typical use |
| ----- | ------------ | ----------- |
| **IdentityAgent** | Pass‑through (output = input). | Simple branching / debugging. |
| **CounterAgent** | Sums integer fields across a list of inputs. Config lists which keys to accumulate. | Stats, metrics aggregation. |
| **DebugAgent** | Prints any incoming object with rich‑console colouring. Sink (no output). | Inspect mid‑pipeline data. |
| **PrintAgent** | Emits formatted console output; can also append to a log file. | “Human readable checkpoint” inside pipeline. |
| **SaveJsonAgent** | Serialises incoming pydantic model to disk (`filename` from config). Optionally includes parent UUID in the filename. | Archiving intermediate results. |
| **LoadJsonAgent** | Reverse of above: reads JSON into the specified `model_class`. | Pipeline warm‑start. |
| **ListCollectionAgent** | Collects partial list outputs that belong to the same parent, merges once all pieces are present. | Stitch chunked LLM responses back together. |
| **MultiPortAggregatorAgent** | Accepts **multiple named input ports**; emits one message after it finds sibling messages that "line up" across ports (heuristic based on parent IDs). | Join data that was produced by several parallel branches.


---

## 5  Message‑ID conventions (`parents`)

* Every outbound port appends **one** ID of the form `uuid:idx:len` where:
  * `uuid`   – random per `send()` call.
  * `idx`    – zero‑based index inside the list produced by the transformer.
  * `len`    – total list length (**1** for non‑list messages).
* `parents` therefore grows like a stack as messages move downstream, allowing advanced agents to:
  * test whether all siblings are present (`idx == len − 1` across all ids),
  * find common prefixes (see utility `longest_common_sublist`).


---

## 6  Serialisation & resume‑after‑crash

* **Agents** store:
  * Arbitrary `_state` (must be pydantic if you want JSON serialisation).
  * Full input‑queue & unconnected outputs.
* **Scheduler** stores its counters and the **order** in which it will visit agents next.  That means you can:

```python
# After a crash …
sched = AgentScheduler(save_dir="snapshots")
sched.add_agent(MyAgent1())
sched.add_agent(MyAgent2())
…
# Re‑hydrate
sched.load_agents("snapshots/step_42")
sched.load_state("snapshots/step_42")
# Continue where you left off
sched.step_all()
```


---

## 7  Extending the system – writing your own agent

```python
class MyFancyAgentConfig(BaseToolConfig):
    threshold: float = 0.7

class MyInput(BaseIOSchema):
    text: str

class MyOutput(BaseIOSchema):
    score: float
    label: str

class MyFancyAgent(ConnectedAgent):
    input_schema = MyInput
    output_schema = MyOutput

    def __init__(self, config: MyFancyAgentConfig, uuid="fancy1"):
        super().__init__(config, uuid)
        self._cfg = config

    def run(self, params: MyInput) -> MyOutput:
        score = my_model(params.text)
        return MyOutput(score=score, label="POS" if score>self._cfg.threshold else "NEG")
```

**Key rules**:
1. Implement **`run()`** (or override `process()` if you need access to `parents`).
2. Use pydantic models for IO – the rest of the framework will then automatically validate, serialise, and pretty‑print.
3. Instantiate the agent, connect its port(s), add it to the scheduler, and go.


---

## 8  Putting it all together – a minimal pipeline

```python
# 1.  Create agents
loader   = LoadJsonAgent(LoadJsonAgentConfig(filename="docs.json", model_class=DocModel), uuid="loader")
segment  = MyFancyAgent(MyFancyAgentConfig(threshold=0.5), uuid="segment")
printer  = PrintAgent(PrintAgentConfig(), uuid="printer")

# 2.  Wire them
loader.connectTo(segment)       # fan‑out not needed here
segment.connectTo(printer)

# 3.  Schedule
sched = AgentScheduler(save_dir="ckpt", save_step=10)
for ag in (loader, segment, printer):
    sched.add_agent(ag)

# 4.  Kick‑off	(press ➡️  and watch the magic)
loader.feed(BaseIOSchema())     # seed – loader ignores the content
sched.step_all()
```

---

## 9  Current limitations & roadmap

| Area | Status | Ideas |
| ---- | ------ | ----- |
| **Async / await** | single‑threaded right now | integrate `asyncio`, per‑agent concurrency controls. |
| **Back‑pressure** | queues grow unbounded | configurable max‑queue + flow‑control feedback. |
| **Port type‑safety** | manual – user must match schemas | optional static check at connect‑time. |
| **Visualization** | none yet | auto‑generate Graphviz / Mermaid diagrams from connections. |
| **Metrics** | basic counts only | Prometheus exporter for queue length, step rate, error count. |

---

_That should give you a solid mental model of how the pieces fit together._

Have fun building!


---

## 10  Pipeline visualisation helper

Sometimes you just want a **quick, human‑readable picture** of how your agents are wired.  Below is a lightweight helper that walks the list you passed to `AgentScheduler` and prints an **ASCII flow diagram**.  

### Usage

```python
    printer = PipelinePrinter(is_ortho=False,
    						  direction='TB',
                              fillcolor='blue',
                              show_schemas=True,
                              schema_fillcolor='moccasin')
    printer.print_ascii(scheduler.agents)
    printer.to_png(scheduler.agents,  'pipeline_bookmarks.png') # if 'dot' available
```

_Output example ‘(ASCII)’:_

```text
LoadJsonAgent#1
  ├─▶ MyFancyAgent#1
  └─▶ DebugAgent#1
MyFancyAgent#1
  └─▶ MultiPortAggregatorAgent#1@text
DebugAgent#1
MultiPortAggregatorAgent#1
  └─▶ PrintAgent#1
```



_Output example ‘(PNG)’:_

---

![Pipeline Condition](pic\pipeline_condition.png)

---

![Pipeline Bookmarks](pic\pipeline_bookmarks.png)

---

![Pipeline News](pic\pipeline_news.png)

---


![Pipeline Condition](pic\pipeline_condition_large.png)

---

![Pipeline Bookmarks](pic\pipeline_bookmarks_large.png)

---

![Pipeline News](pic\pipeline_news_large.png)

---
