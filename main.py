from quex.parser import parse_qasm_string

sample_circuit = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

operations = parse_qasm_string(sample_circuit)
print("Parsed Operations:")
for op in operations:
    print(op)
