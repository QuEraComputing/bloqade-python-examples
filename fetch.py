from bloqade import load
import sys


batch = load(sys.argv[1])
print(batch.tasks_metric())
