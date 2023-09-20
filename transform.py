import json
import os, sys

from bloqade import loads

src = None
with open(sys.argv[1], "r") as f:
    src = json.load(f)


tasks = src["remote_batch"]["tasks"]

new_tasks = []
for tid, task in tasks:
    task_id = task["braket_task"]["task_id"]
    task_ir = task["braket_task"]["task_ir"]["quera_task_specification"]
    task_result_ir = task["braket_task"]["task_result_ir"]["task_result_ir"]
    backend = task["braket_task"]["backend"]["braket_backend"]

    parallel_decoder = (
        None
        if "parallel_decoder" not in task["braket_task"]
        or task["braket_task"]["parallel_decoder"] is None
        else task["braket_task"]["parallel_decoder"]["parallel_decoder"]
    )

    tsk = {
        "bloqade.task.braket.BraketTask": {
            "backend": backend,
            "parallel_decoder": parallel_decoder,
            "task_id": task_id,
            "task_result_ir": task_result_ir,
            "task_ir": task_ir,
            "metadata": task["braket_task"]["metadata"],
        }
    }
    new_tasks.append([tid, tsk])


new_future = {
    "bloqade.task.batch.RemoteBatch": {
        "source": src["remote_batch"]["source"],
        "name": src["remote_batch"]["name"],
        "tasks": new_tasks,
    }
}

a = loads(json.dumps(new_future))

# str = json.dumps(new_future)

with open(sys.argv[1] + ".new", "w") as f:
    json.dump(new_future, f, indent=2)

# print(str)
