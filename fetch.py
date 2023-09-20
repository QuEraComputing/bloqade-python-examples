from bloqade import load, save
import sys


batch = load(sys.argv[1])
batch.fetch()
save(batch, sys.argv[1])