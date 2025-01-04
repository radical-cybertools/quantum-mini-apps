import pennylane as qml 
from pennylane import numpy as np
from pennylane import math
from pennylane import grad
from scipy.optimize import minimize
from time import perf_counter
from pennylane.transforms import merge_rotations, commute_controlled
from mini_apps.qml_data_compression.utils.custom_two_qubit_decomposition import two_qubit_decomposition

from mini_apps.qml_data_compression.utils.custom_single_qubit_unitary import one_qubit_decomposition

# SO(4) and SU(2) x SU(2). For A in SO(4), E A E^\dag is in SU(2) x SU(2).
E = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
Edag = E.conj().T
CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

class UnitaryToPennylane():
    def __init__(self, network) -> None:
        self.network = network

    def _untitary_to_gates(self, RY=True):
        blocks, layers = self.network.shape[:2]
        Lc = (blocks + 1) // 2 - 1

        gates_list, parameters, wires = [], [], []
        # TODO: Move axis and loop over layers instead of indexing with depth
        for layer in range(layers):
            gates = self.network[:, layer]

            gates_right = gates[Lc:]
            for i, gate in enumerate(gates_right):
                idx = Lc + i
                if np.allclose(gate, np.eye(4)):
                    continue
                
                if RY:
                    operations = self._decomposition_2_cnots(gate, [idx, idx+1])
                else:
                    operations = two_qubit_decomposition(gate, [idx, idx+1])
                # TODO: define correct assert for new decomposition
                for operation in operations:
                    gates_list.append(operation._name)
                    # TODO: Make this nicer, check if operation has a parameter
                    if operation._name not in ["CNOT", "S", "SX", "C(PauliX)"]: parameters += operation.parameters
                    wires.append(operation.wires)

            gates_left = gates[Lc-1::-1]
            for i, gate in enumerate(gates_left):
                idx = Lc-1 - i 
                if RY:
                    operations = self._decomposition_2_cnots(gate, [idx, idx+1])
                else:
                    operations = two_qubit_decomposition(gate, [idx, idx+1])
                for operation in operations:
                    gates_list.append(operation._name)
                    # TODO: Make this nicer, check if operation has a parameter
                    if operation._name not in ["CNOT", "S", "SX", "C(PauliX)"]: parameters += operation.parameters
                    wires.append(operation.wires)

        return gates_list, parameters, wires

    def get_circuit(self, RY=False):
        gates, params, wires = self._untitary_to_gates(RY=RY)
        circuit, params = self.generate_qnode_params(gates, params, wires)
        # params = circuit.tape.get_parameters()
        params = np.array(params, requires_grad=True)

        if RY:
            circuit, params = self.generate_qnode_params(gates, params, wires)
            # TODO: Check if this correct
            if "PauliX" in [op.name for op in circuit.tape.operations]:
                wires, gates, params = self.commute_x_gates(circuit.tape.operations)
            circuit = self.generate_qnode_params(gates, params, wires)
            gates, params, wires = self.compile_circuit(circuit.func, params)
            circuit, params = self.generate_qnode_params(gates, params, wires)
        else:
            gates, params, wires = self.compile_circuit_old(circuit.func, params)
            circuit, params = self.generate_qnode_params(gates, params, wires)

        params = params % (2 * np.pi)
        return circuit, params.flatten()

    def generate_qnode_params(self, gates, params, wires, device="default.qubit"):
        dev = qml.device(device, wires=len(self.network) + 1 )
        @qml.qnode(dev)
        def circuit(params):
            counter = 0
            for gate, wire in zip(gates, wires):

                if gate == "RX":
                    qml.RX(params[counter], wire)
                    counter += 1

                elif gate == "RY":
                    qml.RY(params[counter], wire)
                    counter += 1

                elif gate == "RZ":
                    qml.RZ(params[counter], wire)
                    counter += 1

                elif gate == "Rot":
                    qml.Rot(params[counter], params[counter+1], params[counter+2], wire)
                    counter += 3

                elif gate == "CNOT" or gate == "C(PauliX)":
                    qml.CNOT(wire)

                elif gate == "S":
                    qml.S(wire)

                elif gate == "SX":
                    qml.SX(wire)

                elif gate == "PauliX":
                    qml.PauliX(wire)

                elif gate == "PauliZ":
                    qml.PauliZ(wire)

                elif gate == "Barrier":
                    qml.Barrier(wire)

                else:
                    raise NotImplementedError(f"Gate {gate} not implemented")

            return qml.state()

        params = np.array(params, requires_grad=True)
        # circuit(params)

        return circuit, params

    @staticmethod
    def commute_x_gates(operations):

        # Step 1: Commute X gates to the beginning of the circuit
        final_commution = False
        while not final_commution:

            new_operations = []
            found_first_x = False
            for i, op in enumerate(operations[::-1]):

                if op.name == "PauliX" and not found_first_x:
                    found_first_x = True
                    wire = op.wires[0]
                elif found_first_x and op.name == "RY" and op.wires[0] == wire:
                    new_operations.append(qml.RY(-op.parameters[0], wires=op.wires))
                elif found_first_x and op.name == "CNOT" and op.wires[0] == wire:
                    new_operations.append(qml.CNOT(op.wires))
                    new_operations.append(qml.PauliX(op.wires[1]))
                elif found_first_x and op.name == "CNOT" and op.wires[1] == wire:
                    new_operations.append(qml.CNOT(op.wires))
                elif found_first_x and op.name == "PauliX" and op.wires[0] == wire:
                    new_operations += operations[::-1][i+1:]
                    break
                elif i == len(operations[::-1]) - 1:
                    new_operations.append(op)
                    new_operations.append(qml.PauliX(wire))
                    break # new
                else:
                    new_operations.append(op)

            operations = new_operations[::-1]

            # Check if there is an X gate that it is the first gate on a wire
            all_true = np.ones(13, dtype=bool)
            for i, op in enumerate(operations):
                if op.name == "PauliX":
                    wire = op.wires[0]
                    for j, op2 in enumerate(operations[:i]):
                        if wire in op2.wires:
                            all_true[wire] = False
                            break
            final_commution = np.all(all_true)

        # Step 2: Merge the first X gates with the RY gates
        for i, op in enumerate(operations):
            if op.name == "PauliX":
                wire = op.wires
                for j , op2 in enumerate(operations[i+1:]):
                    if op2.wires == wire and op2.name == "RY":
                        phi = (op2.data[0] + np.pi) % (2 * np.pi)
                        operations[i+j+1] = qml.RY(phi, op2.wires)
                        break

        # Step 3: remove the leading X gates and extract the wires and gates
        operations = [op for op in operations if op.name != "PauliX"]

        wires = [op.wires for op in operations]
        gates = [op.name for op in operations]
        params = [op.data for op in operations if op.data != ()]

        return wires, gates, params

    @staticmethod
    def compile_circuit(circuit, weights):
        circuit = commute_controlled(circuit, direction='left')
        circuit = merge_rotations(circuit, atol=0)
        tape = qml.transforms.make_tape(circuit)(weights)

        remove_first_rz = [False] * tape.num_wires
        operation_list, paramter_list, wires_list = [], [], []
        for op in tape.circuit[:-1]:
            if op.name == "Rot" and remove_first_rz[op.wires[0]]:
                operation_list += ["RY", "RZ"]
                wires_list += [op.wires] * 2
                paramter_list += op.parameters[1:]
                remove_first_rz[op.wires[0]] = False
            else:
                operation_list.append(op.name)
                paramter_list.extend(op.parameters)
                wires_list.append(op.wires)

        paramter_list = np.array(paramter_list, requires_grad=True)

        return operation_list, paramter_list, wires_list


    @staticmethod
    def compile_circuit_old(circuit, weights):
        circuit = commute_controlled(circuit, direction='left')
        circuit = merge_rotations(circuit, atol=0)
        tape = qml.transforms.make_tape(circuit)(weights)

        remove_first_rz = [True] * tape.num_wires

        operation_list, paramter_list, wires_list = [], [], []
        for op in tape.circuit[:-1]:
            if op.name == "Rot" and remove_first_rz[op.wires[0]]:
                operation_list += ["RY", "RZ"]
                wires_list += [op.wires] * 2
                paramter_list += op.parameters[1:]
                remove_first_rz[op.wires[0]] = False
            else:
                operation_list.append(op.name)
                paramter_list.extend(op.parameters)
                wires_list.append(op.wires)

        paramter_list = np.array(paramter_list, requires_grad=True)

        return operation_list, paramter_list, wires_list


    @staticmethod
    def train_circuit(circuit, target, params, use_jit=False, maxiter=5):
        params = np.array(params, requires_grad=True)

        # if use_jit:
        #     # TODO: remove double def of loss function
        #     import jax
        #     jax.config.update("jax_enable_x64", True)
        #     from jax import numpy as jnp
        #     from jax import jit, grad
        #     circuit = jit(circuit)
        #     params = jnp.array(params)
        #     target = jnp.array(target)

        #     def cost_fn(params):
        #         return 1 - jnp.sum(jnp.abs(target.conj() * circuit(params)))

        #     circuit = jit(circuit)
        #     cost_fn = jit(cost_fn)
        #     grad_fn = jit(grad(cost_fn))
        # else:
        
        def cost_fn(params):
            return 1 - np.sum(np.abs(target.conj() * circuit(params)))
        grad_fn = grad(cost_fn, argnum=0)

        class Callback():
            def __init__(self):
                self.iteration = 0
                self.losses= []
                self.tic = perf_counter()

            def callback(self, xk):
                # print(f"Epoch {self.iteration}: {1 - cost_fn(xk)}")
                self.iteration += 1
                self.losses.append(float(1 - cost_fn(xk)))

            def ellapsed_time(self):
                return perf_counter() - self.tic


        # start = perf_counter()
        # cost_fn(params)
        # print("Elapsed time warm up loss: ", perf_counter() - start)
        # start = perf_counter()
        # grad_fn(params)
        # print("Elapsed time warm up grad: ", perf_counter() - start)

        callback = Callback()
        res = minimize(cost_fn, params, method='BFGS', jac=grad_fn, callback=callback.callback, options={"return_all": True, "gtol": 1e-20, 'maxiter': maxiter})

        return res, callback.losses

    @staticmethod
    def wire_phi_AB_CD(U, wire_phi, phi, delta, wires):

        wire_delta = (wire_phi + 1) % 2

        interior_decomp = [
            qml.CNOT(wires=[wires[1], wires[0]]),
            qml.RY(delta, wires=wires[wire_delta]),
            qml.RY(phi, wires=wires[wire_phi]),
            qml.CNOT(wires=[wires[1], wires[0]]),
        ]

        RY_a = qml.RY(math.cast_like(delta, 1j), wires=0).matrix()
        RY_b = qml.RY(phi, wires=0).matrix()

        inner_matrix = math.kron(RY_a, RY_b) if wire_delta == 0 else math.kron(RY_b, RY_a)
        V = CNOT10 @ inner_matrix @ CNOT10

        u = Edag @ U @ E
        v = Edag @ V @ E
        uuT = math.dot(u, math.T(u))
        vvT = math.dot(v, math.T(v))
        _, p = math.linalg.eigh(math.real(uuT) + math.imag(uuT))
        _, q = math.linalg.eigh(math.real(vvT) + math.imag(vvT))
        p = p @ math.diag([1, 1, 1, math.sign(math.linalg.det(p))])
        q = q @ math.diag([1, 1, 1, math.sign(math.linalg.det(q))])
        G = p @ math.T(q)
        H = math.dot(math.conj(math.T(v)), math.dot(math.T(G), u))
        AB = E @ G @ Edag
        CD = E @ H @ Edag
        boolean = np.allclose(np.imag(AB), np.zeros([4, 4])), np.allclose(np.real(AB), np.zeros([4, 4]))

        return AB, CD, interior_decomp, boolean

    @staticmethod
    def _su2su2_to_tensor_products(U):

        C1 = U[0:2, 0:2]
        C2 = U[0:2, 2:4]
        C3 = U[2:4, 0:2]
        C4 = U[2:4, 2:4]

        C14 = math.dot(C1, math.conj(math.T(C4)))
        a1 = math.sqrt(math.cast_like(C14[0, 0], 1j))

        C23 = math.dot(C2, math.conj(math.T(C3)))
        a2 = math.sqrt(-math.cast_like(C23[0, 0], 1j))
        C12 = math.dot(C1, math.conj(math.T(C2)))

        if not math.allclose(a1 * math.conj(a2), C12[0, 0]):
            a2 *= -1

        A = math.stack([math.stack([a1, a2]), math.stack([-math.conj(a2), math.conj(a1)])])
        use_B2 = math.allclose(A[0, 0], 0.0, atol=1e-6)
        B = C2 / math.cast_like(A[0, 1], 1j) if use_B2 else C1 / math.cast_like(A[0, 0], 1j)

        return math.convert_like(A, U), math.convert_like(B, U)

    def _decomposition_2_cnots(self, U, wires):
        u = math.dot(Edag, math.dot(U, E))
        gammaU = math.dot(u, math.T(u))
        evs, _ = math.linalg.eig(gammaU)

        x = math.angle(evs[0])
        y = math.angle(evs[1])
        if math.allclose(x, -y):
            y = math.angle(evs[2])

        phi = np.abs(x - y) / 2 % np.pi
        delta = np.abs(x + y) / 2  % np.pi

        for wire_phi in [0, 1]:
            AB, CD, interior_decomp, boolean = self.wire_phi_AB_CD(U, wire_phi, phi, delta, wires)

            A, B = self._su2su2_to_tensor_products(AB)
            C, D = self._su2su2_to_tensor_products(CD)

            if True in boolean:
                if np.abs(np.imag(A[0, 0])) > np.abs(np.real(A[0, 0])):
                    A = np.imag(A)

                if np.abs(np.imag( B[0, 0])) > np.abs(np.real( B[0, 0])):
                    B = np.imag(B)

                if np.abs(np.imag(C[0, 0])) > np.abs(np.real(C[0, 0])):
                    C = np.imag(C)

                if np.abs(np.imag(D[0, 0])) > np.abs(np.real(D[0, 0])):
                    D = np.imag(D)

                assert np.allclose(np.linalg.det(A) * np.linalg.det(B) * np.linalg.det(C) * np.linalg.det(D), 1)

                break

        if np.allclose(np.linalg.det(A), 1):
            angle_a = 2 * np.arcsin(np.real(A)[1, 0])
            A_ops = [qml.RY(float(angle_a), wires=wires[0])]
        else:
            A_ops = one_qubit_decomposition(A, wires[0], rotations='XYX')

        if np.allclose(np.linalg.det(B), 1):
            if B[1,0] < 0: B = -B
            angle_b = 2 * np.arccos(np.real(B)[1, 1])
            B_ops = [qml.RY(float(angle_b), wires=wires[1])]
        else:
            B_ops = one_qubit_decomposition(B, wires[1], rotations='XYX')

        if np.allclose(np.linalg.det(C), 1):
            angle_c = 2 * np.arcsin(np.real(C)[1, 0])
            C_ops = [qml.RY(float(angle_c), wires=wires[0])]
        else:
            C_ops = one_qubit_decomposition(C, wires[0], rotations='XYX')

        if np.allclose(np.linalg.det(D), 1):
            if D[0,0] < 0: D = -D
            angle_d = 2 * np.arcsin(np.real(D)[1, 0])
            D_ops = [qml.RY(float(angle_d), wires=wires[1])]
        else:
            D_ops = one_qubit_decomposition(D, wires[1], rotations='XYX')

        A_ops = self.transform_rz_to_pauli(A_ops)
        B_ops = self.transform_rz_to_pauli(B_ops)
        C_ops = self.transform_rz_to_pauli(C_ops)
        D_ops = self.transform_rz_to_pauli(D_ops)

        return C_ops + D_ops + interior_decomp + A_ops + B_ops


    @staticmethod
    def transform_rz_to_pauli(gate_list):
        new_gate_list = []
        for gate in gate_list:
            # TODO: make this stable, +1e-10 is used for mod
            angle = gate.data[0] + 1e-10
            if gate._name == "RZ":
                if np.allclose((angle / np.pi) % 2, 0):
                    continue
                elif np.allclose((angle / np.pi) % 1, 0):
                    new_gate_list.append(qml.PauliZ(wires=gate._wires[0]))
                else:
                    print(f"Problem {angle}")

            elif gate._name == "RX":
                if np.allclose((angle / np.pi) % 2, 0):
                    continue
                elif np.allclose((angle / np.pi) % 1, 0):
                    new_gate_list.append(qml.PauliX(wires=gate._wires[0]))
                else:
                    print(f"Problem {angle}")
            else:
                new_gate_list.append(gate)
        return new_gate_list


def check_pauliX_at_beginning_per_wire(operations, num_wires):
    pauliX_at_beginning_per_wire = {wire: True for wire in range(num_wires)}

    for op in operations:
        print(pauliX_at_beginning_per_wire)
        if op.name == 'PauliX':
            for wire in op.wires:
                if not pauliX_at_beginning_per_wire[wire]:
                    return False
        else:
            for wire in op.wires:
                pauliX_at_beginning_per_wire[wire] = False
    
    return True

