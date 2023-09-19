from bloqade import start, var, load, save

# from bloqade.task import RemoteBatch
from bloqade.atom_arrangement import Chain
import numpy as np
import os
import matplotlib.pyplot as plt

delta = 0.377371  # Detuning
xi = 3.90242  # Phase jump
tau = 4.29268  # Time of each pulse

amplitude_max = 10
min_time_step = 0.05
detuning_value = delta * amplitude_max
T = tau / amplitude_max - min_time_step

durations = [
    min_time_step,
    T,
    min_time_step,
    min_time_step,
    min_time_step,
    T,
    min_time_step,
]
rabi_wf_values = [
    0.0,
    amplitude_max,
    amplitude_max,
    0.0,
    0.0,
    amplitude_max,
    amplitude_max,
    0.0,
]

run_time = var("run_time")

lp_gate_sequence = (
    start.rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_wf_values)
    .slice(0, run_time - min_time_step)
    .record("rabi_value")  # record the rabi value at the end of the slice
    .linear("rabi_value", 0, min_time_step)  # add a linear ramp to zero
    .detuning.uniform.constant(detuning_value, sum(durations))
    .slice(0, run_time)
    .phase.uniform.piecewise_constant(durations, [0.0] * 4 + [xi] * 3)
    .slice(0, run_time)
    .parse_sequence()
)

run_times = np.arange(min_time_step, sum(durations), min_time_step)

# create a one atom and two atom example
geometries = {"1-atom": Chain(1), "2-atom": Chain(2, 4.0)}

batches = {
    name: geometry.apply(lp_gate_sequence).batch_assign(run_time=run_times)
    for name, geometry in geometries.items()
}


emu_filesnames = {}
for name, batch in batches.items():
    emu_filename = os.path.join(
        os.path.abspath(""), "data", f"lp-gate-{name}-emulation.json"
    )
    emu_filesnames[name] = emu_filename

    if not os.path.isfile(emu_filename):
        emu_batch = batch.braket.local_emulator().run(shots=10000)

        save(emu_batch, emu_filename)


filenames = {}
for name, batch in batches.items():
    filename = os.path.join(os.path.abspath(""), "data", f"lp-gate-{name}-job.json")
    filenames[name] = filename

    if not os.path.isfile(filename):
        hardware_batches = (
            batch.parallelize(24)
            .braket.aquila()
            .run_async(shots=100, ignore_submission_error=True)
        )
        save(hardware_batches, filename, indent=2)


# %%
emu_batches = {name: load(filename) for name, filename in emu_filesnames.items()}
hardware_batches = {name: load(filename) for name, filename in filenames.items()}
# for name, filename in filenames.items():
#     hardware_batches[name].fetch()
#     save(filename, hardware_batches[name])


# %% [markdown]


# %%
emu_lines = []
for name, batch in emu_batches.items():
    report = batch.report()
    run_times = report.list_param("run_time")
    ground_state_atoms = [
        (1 - bitstring).mean(axis=0).sum() for bitstring in report.bitstrings()
    ]

    (ln,) = plt.plot(run_times, ground_state_atoms, color="#878787")
    ln.set_label("Emulation")
    emu_lines.append(ln)

colors = ["#55DE79", "#EDFF1A"]
hw_lines = []
for color, (name, batch) in zip(colors, hardware_batches.items()):
    report = batch.report()
    run_times = report.list_param("run_time")
    ground_state_atoms = [
        (1 - bitstring).mean(axis=0).sum() for bitstring in report.bitstrings()
    ]

    (ln,) = plt.plot(run_times, ground_state_atoms, color=color, label=name)
    hw_lines.append(ln)

plt.legend(handles=[*hw_lines, emu_lines[0]])
plt.xlabel("Time ($\mu s$)")
plt.ylabel(r"Atoms in $|g\rangle$")
plt.show()
