import json
import os,sys

src = None
with open(sys.argv[1],"r") as f:
	src = json.load(f)


tasks = src["hardware_task_shot_results"]


new_future={"remote_batch": 
				{"source": None, 
				 "name": None, 
				 "tasks": []
				}
			}


for tid, task in tasks.items():
	task_id = task["task_id"]
	task_ir = task["hardware_task"]["task_ir"]
	task_result_ir = task["task_result_ir"]
	backend = task["hardware_task"]["braket_backend"]	

	parallel_decoder = None if "parallel_decoder" not in task["hardware_task"] else task["hardware_task"]["parallel_decoder"]
	
	tsk = ({"braket_task": 
		{
			"backend": {"braket_backend": backend },
			"parallel_decoder": parallel_decoder,
			"task_id": task_id,
			"task_result_ir": {"task_result_ir": task_result_ir},
			"task_ir": {"quera_task_specification": task_ir},
			"metadata": {}
		}
	})


	new_future["remote_batch"]["tasks"].append([int(tid),tsk])



str = json.dumps(new_future)
	
with open(sys.argv[1]+".new","w") as f:
	f.write(str)

#print(str)











