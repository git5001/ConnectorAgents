from AgentFramework.core.AgentScheduler import AgentScheduler
from AgentFramework.core.IdentityAgent import IdentityAgent
from AgentOrchestrator.test.ToolChoosingOrchestrator import ToolChoosingOrchestrator, TaskRequest, ConsolePrinterAgent

# ---- quick manual demo -----------------------------------------------------

if __name__ == "__main__":
    sched = AgentScheduler()


    orch = ToolChoosingOrchestrator()
    sched.add_agent(orch)

    cons = ConsolePrinterAgent()
    sched.add_agent(cons)
    orch.connectTo(cons)

    iden = IdentityAgent()
    sched.add_agent(iden)
    orch.connectTo(iden)
    iden = IdentityAgent()
    sched.add_agent(iden)
    orch.connectTo(iden)

    print("*"*100)
    orch.feed(TaskRequest(prompt="Please read file demo"))
    sched.step_all(True)
    print("OUT1", iden.pop_one_output())
    print("*"*100)

    orch.feed(TaskRequest(prompt="Fetch web page"))
    sched.step_all(True)
    print("OUT2",sched.pop_one_output_for_agent(iden))
    print("*"*100)

    orch.feed(TaskRequest(prompt="Run python hello world"))
    sched.step_all(True)
    for output in sched.pop_all_outputs():
        print("OUT3",output)
    print("*"*100)

