import quex as qx
import numpy as np

import time

sample_circuit = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

print("Below is an openqasm circuit:\n" + 
      3 * "-" + f"\n{sample_circuit}" + 3 * "-")

print("This is circuit object from openqasm")
qc = qx.Circuit.from_qasm(sample_circuit)
print(qc)

for i, d in enumerate(qc.layers):
    print(f"Depth: {i}, Operations: {d}")

print(f"Now lets generate random circuit using {qx.random_qasm.__name__}")

c = qx.random_qasm(num_qubits=12, depth=12)
print(c)

print("This can be used to construct Circuit")

qc = qx.Circuit.from_qasm(c)

print("The circuit")

print(qc)

print("\nFlat operations:\n")
for i, d in enumerate(qc.operations):
    print(f"Depth: {i}, Operations: {d}")

print("Grouped operations:\n")
for i, d in enumerate(qc.layers):
    print(f"Depth: {i}, Operations: {d}")



tic = time.time()
# 1. Build a random circuit
qasm_str = qx.random_qasm(3, 12)

qc = qx.Circuit.from_qasm(qasm_str)
print("Circuit:")
print(qc)
print("-" * 30)

# 2. Run the Simulation!
sim = qx.NumpySimulator()
final_state_tensor = sim.run(qc)

# 3. Flatten the N-dimensional tensor to a standard 1D column vector for viewing
state_vector = final_state_tensor.flatten()

# Calculate probabilities (Magnitude squared: |psi|^2)
probabilities = np.abs(state_vector) ** 2
toc = time.time()

print("Final State Vector:")
for i, amp in enumerate(state_vector):
    # Convert integer 'i' to binary string to represent the qubit state (e.g., 0 -> '00')
    state_label = format(i, f'0{qc.num_qubits}b')
    print(f"|{state_label}>: Amplitude = {amp.real:18.10f} + {amp.imag:18.10f}i  | Prob = {probabilities[i]:18.10%}")

print(f"Time in Numpy simulator: {toc - tic}")

# A larger circuit
tic = time.time()
# 1. Build a random circuit
qasm_str = qx.random_qasm(18, 120)

qc = qx.Circuit.from_qasm(qasm_str)
print("Circuit:")
print(qc)
print("-" * 30)

# 2. Run the Simulation!
sim = qx.NumpySimulator()
final_state_tensor = sim.run(qc)

# 3. Flatten the N-dimensional tensor to a standard 1D column vector for viewing
state_vector = final_state_tensor.flatten()

# Calculate probabilities (Magnitude squared: |psi|^2)
probabilities = np.abs(state_vector) ** 2
max_index = np.argmax(probabilities)
toc = time.time()

result_state = format(max_index, f'0{qc.num_qubits}b')
print(f"Most probable state: |{result_state}>")
print(f"Time in Numpy simulator: {toc - tic}")


# Build the circuit
qc = qx.Circuit(num_qubits=2)
qc.add_operation('h', 0)
qc.add_operation('rx', 0, params=["t0"]) # Use a string!
qc.add_operation('cx', [0, 1])
qc.add_operation('ry', 1, params=["t1"]) # Use a string!

print("Static Circuit Template:")
print(qc)

sim = qx.NumpySimulator()

# 2. The High-Speed ML Optimization Loop
start_time = time.time()
iterations = 1000

for i in range(iterations):
    # The optimizer generates new angles...
    new_angles = {
        "t0": 0.5 + (i * 0.001),
        "t1": 1.2 - (i * 0.001)
    }
    
    # Execute instantly by passing the dictionary directly to the backend
    state = sim.run(qc, parameter_binds=new_angles)

end_time = time.time()
print(f"\nExecuted {iterations} circuits in {end_time - start_time:.4f} seconds!")