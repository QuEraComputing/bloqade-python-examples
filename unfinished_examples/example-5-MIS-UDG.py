from bloqade.ir.location import Square
import numpy as np
import bloqade

# durations for rabi and detuning
durations = [0.3, 1.6, 0.3]

mis_udg_program = (
    Square(15, 5.0)
    .apply_defect_density(0.3)
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, [0.0, 15.0, 15.0, 0.0])
    .detuning.uniform.piecewise_linear(
        durations, [-30, -30, "final_detuning", "final_detuning"]
    )
)

mis_udg_job = mis_udg_program.batch_assign(final_detuning=np.linspace(0, 80, 81))
# submit to HW
hw_batch = mis_udg_job.braket.aquila().submit(shots=1000)

bloqade.save("example-5-MIS-UDG.json",hw_batch)

# submit to emulator would take too many resources
