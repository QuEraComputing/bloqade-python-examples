from bloqade import load, save

file = "data/quantum-scar-dynamics-job.json"
obj = load(file)
obj.fetch()

save(obj, "data/quantum-scar-dynamics-job.json")