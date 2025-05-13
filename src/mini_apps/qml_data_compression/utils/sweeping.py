import numpy as np
import scipy as sp
from time import perf_counter

def random_unitaries(shape, strength=1e-2, orthogonal=False):
    if orthogonal:
        unitaries = np.random.normal(0, strength, size=shape)
    else:
        unitaries = np.random.normal(0, strength, size=shape)\
        + 1j * np.random.normal(0, strength, size=shape)
    unitaries += np.eye(shape[-2], shape[-1])
    unitaries, rs = np.linalg.qr(unitaries)
    unitaries *= np.sign(np.einsum('...ii->...i', rs))[...,None,:]
    return unitaries

class staircase_circuit:
    """
    Optimize the overlap of a shallow staircase circuit
    with an MPS by sweeping through the gates.
    
    """
    def __init__(self,
                 L,
                 layers,
                 d=2,
                 batchsize=1,
                 initial_gates=None,
                 orthogonal=False,
                 Lc=None,
                 gate_layout=None):

        # TODO: Add for sparse circuits as well
        self.L = L
        if Lc is None:
            self.Lc = self.L//2-1
        else:
            self.Lc = Lc

        self.d = d
        self.layers = layers
        self.batch = batchsize
        # parameters for contraction
        self.svd_min = 1e-10
        self.chi_min = 32
        self.chi_max = 512
        # initial_gates
        strength = 1e-10
        if initial_gates is None:
            self.gates = random_unitaries(
                shape=(self.batch, self.L-1, self.layers, self.d**2, self.d**2),
                strength=strength,
                orthogonal=orthogonal)
        else:
            gates = initial_gates[:self.batch, :self.L-1, :self.layers]
            batch, L, layers, d1, d2 = initial_gates.shape
            assert d1 == d2 and d2 == self.d**2
            if batch < self.batch:
                gates = np.append(
                    gates,
                    random_unitaries(
                        shape=(self.batch-batch, L, layers, d2, d2),
                        strength=strength),
                        orthogonal=orthogonal,
                        axis=0)
            if L < self.L-1:
                gates = np.append(
                    gates,
                    random_unitaries(
                        shape=(self.batch, self.L-1-L, layers, d2, d2),
                        strength=strength),
                        orthogonal=orthogonal,
                        axis=1)
            if layers < self.layers+1:
                gates = np.append(
                    gates,
                    random_unitaries(
                        shape=(self.batch, self.L-1, self.layers-layers, d2, d2),
                        strength=strength),
                        orthogonal=orthogonal,
                        axis=2)
            self.gates = gates
        # gate layout
        if gate_layout is None:
            self.gate_layout = np.ones((self.L-1, self.layers), dtype=np.int32)
        else:
            self.gate_layout = gate_layout
        self.gates[:, np.logical_not(self.gate_layout)] = np.eye(self.d**2)

    def left_canonical(self, Bs):
        As = []
        R = np.ones((self.batch, 1, 1))
        for B in Bs:
            _, chil, d, chir = B.shape
            A = np.einsum('iab,ibjc->iajc', R, B)\
                .reshape(self.batch, chil * d, chir)
            Q, R = np.linalg.qr(A)
            As.append(Q.reshape(self.batch, chil, d, chir))
        return As
    
    def right_canonical(self, As):
        Bs = []
        Lt = np.ones((self.batch, 1, 1))
        for A in As[::-1]:
            _, chil, d, chir = A.shape
            A = np.einsum('iajb,icb->iajc', A, Lt)\
                .reshape(self.batch, chil, d * chir)
            Qt, Lt = np.linalg.qr(A.transpose(0,2,1))
            Bs.append(Qt.transpose(0,2,1).reshape(self.batch, chil, d, chir))
        return Bs[::-1]
    
    def apply_gate(self, gate, Al, Ar, which='CB'):
        # reshape gate
        gate = gate.reshape(self.batch, *(self.d,)*4)
        # calculate theta tensor and bring into right shape
        theta = np.einsum('iajb,ibkc->iajkc', Al, Ar)
        _, chil, _, _, chir = theta.shape
        theta = np.einsum('ijklm,ialmb->iajkb', gate, theta)
        theta = theta.reshape(self.batch, chil * self.d, self.d * chir)
        # SVD decomposition and truncation

        # scipy svd does not support batched svd, numpy svd raises error for lapack_driver gesdd
        matrix_shape = (chil * self.d, self.d * chir)
        U, S, V = [], [], []
        for i, matrix in enumerate(theta):
            # U, S, Vt = svd(theta, lapack_driver='gesvd')
            U_iter, S_iter, V_iter = sp.linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd")
            U.append(U_iter)
            S.append(S_iter)
            V.append(V_iter)

        U, S, V = np.stack(U), np.stack(S), np.stack(V)

        chi_new = min(S.shape[1],
                      max(self.chi_min, max(np.sum(S > self.svd_min, axis=1))),
                      self.chi_max)
        if which == 'CB':
            C = (U[:, :, :chi_new] * S[:, None, :chi_new])\
                .reshape(self.batch, chil, self.d, chi_new)
            B = V[:, :chi_new, :].reshape(self.batch, chi_new, self.d, chir)
            return C, B
        elif which == 'AC':
            A = U[:, :, :chi_new].reshape(self.batch, chil, self.d, chi_new)
            C = (S[:, :chi_new, None] * V[:, :chi_new, :])\
                .reshape(self.batch, chi_new, self.d, chir)
            return A, C
        else:
            raise ValueError("Wrong argument 'which' given for 'apply_gate'.")
    
    def _calculate_upper_environments(self, target_Bs):
        upper_envs = []
        As = target_Bs.copy()
        As = self.left_canonical(As)
        # loop through the layers from top to bottom
        for layer in range(self.layers-1, -1, -1):
            # first half sweep from right to left
            gates = np.moveaxis(self.gates[:, :self.Lc:-1, layer], 1, 0)
            for i, gate in enumerate(gates):
                idx = self.L-1-i
                C, B = self.apply_gate(gate.transpose(0,2,1),
                                       As[idx-1],
                                       As[idx],
                                       which='CB')
                As[idx-1] = C
                As[idx] = B
            # second half sweep from left to right
            As = self.right_canonical(As)
            gates = np.moveaxis(self.gates[:, :self.Lc+1, layer], 1, 0)
            for i, gate in enumerate(gates):
                A, C = self.apply_gate(gate.transpose(0,2,1),
                                       As[i],
                                       As[i+1],
                                       which='AC')
                As[i] = A
                As[i+1] = C
            As = self.left_canonical(As)
            upper_envs.append(As.copy())
        # reorder upper_envs before returning
        upper_envs = upper_envs[::-1]
        return upper_envs
    
    def _calculate_right_environments(self, Bs, upper_env):
        right_env = np.ones((self.batch, 1, 1))
        right_envs = [right_env]
        # sweep right to left
        for B, Benv in zip(Bs[:self.Lc+1:-1], upper_env[:self.Lc+1:-1]):
            right_env = np.einsum('iajb,ibc->iajc', Benv, right_env)
            right_env = np.einsum('iajb,icjb->iac', right_env, B)
            right_envs.append(right_env)
        # reorder right_envs before returning
        right_envs = right_envs[::-1]
        return right_envs
    
    def _calculate_left_environment(self, Bs, upper_env):
        left_env = np.ones((self.batch, 1, 1))
        # sweep left to right
        for B, Benv  in zip(Bs[:self.Lc], upper_env[:self.Lc]):
            left_env = np.einsum('iab,iajc->ibjc', left_env, Benv)
            left_env = np.einsum('iajb,iajc->ibc', left_env, B)
        left_env = np.einsum('iab,iajc->ibjc', left_env, upper_env[self.Lc])
        left_env = np.einsum('iajb,iakc->ijbkc', left_env, Bs[self.Lc])
        return left_env
    
    def _calculate_left_environments(self, Bs, upper_env):
        left_env = np.ones((self.batch, 1, 1))
        left_envs = [left_env]
        # sweep left to right
        for B, Benv in zip(Bs[:self.Lc-1], upper_env[:self.Lc-1]):
            left_env = np.einsum('iab,iajc->ibjc', left_env, Benv)
            left_env = np.einsum('iajb,iajc->ibc', left_env, B)
            left_envs.append(left_env)
        # reorder right_envs before returning
        left_envs = left_envs[::-1]
        return left_envs
    
    def _calculate_right_environment(self, Bs, upper_env):
        right_env = np.ones((self.batch, 1, 1))
        # sweep right to left
        for B, Benv in zip(Bs[:self.Lc:-1], upper_env[:self.Lc:-1]):
            right_env = np.einsum('iajb,ibc->iajc', Benv, right_env)
            right_env = np.einsum('iajb,icjb->iac', right_env, B)
        right_env = np.einsum('iajb,ibc->iajc', upper_env[self.Lc], right_env)
        right_env = np.einsum('iajb,ickb->iajck', right_env, Bs[self.Lc])
        return right_env
    
    def _update_gate(self, environment):

        environment = environment.reshape(self.batch, self.d**2, self.d**2)
        # scipy svd does not support batched svd, numpy svd raises error for lapack_driver gesdd
        matrix_shape = (self.d**2, self.d**2)
        X, S, Y = [], [], []
        for i, matrix in enumerate(environment):
            # U, S, Vt = svd(theta, lapack_driver='gesvd')
            X_iter, S_iter, Y_iter = sp.linalg.svd(matrix, lapack_driver="gesvd")
            X.append(X_iter)
            S.append(S_iter)
            Y.append(Y_iter)
        X, S, Y = np.stack(X), np.stack(S), np.stack(Y)

        # X, S, Y = sp.linalg.svd(environment, lapack_driver="gesvd")
        U = X.conj() @ Y.conj()
        det = np.linalg.det(U)
        det = np.array(det)[:, np.newaxis]
        Y[:, -1, :] *= det
        U = X.conj() @ Y.conj()
        return U
    
    def apply_one_layer_right(self, layer, Bs):
        # assumes the given Bs are in right-canonical form
        As = Bs.copy()
        # bring left half into left-canonical form
        R = np.ones((self.batch, 1, 1))
        for i in range(self.Lc):
            _, chil, d, chir = As[i].shape
            A = np.einsum('iab,ibjc->iajc', R, As[i])\
                .reshape(self.batch, chil * d, chir)
            Q, R = np.linalg.qr(A)
            As[i] = Q.reshape(self.batch, chil, d, chir)
        As[self.Lc] = np.einsum('iab,ibjc->iajc', R, As[self.Lc])
        # apply gates on right half
        gates = np.moveaxis(self.gates[:, self.Lc:, layer], 1, 0)
        for i, gate in enumerate(gates):
            idx = self.Lc + i
            A, C = self.apply_gate(gate, As[idx], As[idx+1], which='AC')
            As[idx] = A
            As[idx+1] = C
        As[-1] /= np.linalg.norm(As[-1].reshape(self.batch, -1),
                                 axis=1)[:, None, None, None]
        return As
    
    def apply_one_layer_right_conj(self, layer, Bs):
        # assumes the given Bs are in left-canonical form
        As = Bs.copy()
        # bring right half into right-canonical form
        Lt = np.ones((self.batch, 1, 1))
        for i in range(self.L-1, self.Lc+1, -1):
            _, chil, d, chir = As[i].shape
            A = np.einsum('iajb,icb->iajc', As[i], Lt)\
                .reshape(self.batch, chil, d * chir)
            Qt, Lt = np.linalg.qr(A.transpose(0,2,1))
            As[i] = Qt.transpose(0,2,1).reshape(self.batch, chil, d, chir)
        As[self.Lc+1] = np.einsum('iajb,icb->iajc', As[self.Lc+1], Lt)
        # apply gates on right half
        gates = np.moveaxis(self.gates[:, self.Lc:, layer], 1, 0)
        for i, gate in enumerate(gates):
            idx = self.Lc + i
            A, C = self.apply_gate(gate.conj(), As[idx], As[idx+1], which='AC')
            As[idx] = A
            As[idx+1] = C
        As[-1] /= np.linalg.norm(As[-1].reshape(self.batch, -1),
                                 axis=1)[:, None, None, None]
        return As
    
    def apply_one_layer_left(self, layer, Bs):
        # assumes the given Bs are in left-canonical form
        As = Bs.copy()
        # bring right half into right-canonical form
        Lt = np.ones((self.batch, 1, 1))
        for i in range(self.L-1, self.Lc, -1):
            _, chil, d, chir = As[i].shape
            A = np.einsum('iajb,icb->iajc', As[i], Lt)\
                .reshape(self.batch, chil, d * chir)
            Qt, Lt = np.linalg.qr(A.transpose(0,2,1))
            As[i] = Qt.transpose(0,2,1).reshape(self.batch, chil, d, chir)
        As[self.Lc] = np.einsum('iajb,icb->iajc', As[self.Lc], Lt)
        # apply gates on left half
        gates = np.moveaxis(self.gates[:, self.Lc-1::-1, layer], 1, 0)
        for i, gate in enumerate(gates):
            idx = self.Lc-1 - i
            C, B = self.apply_gate(gate, As[idx], As[idx+1], which='CB')
            As[idx] = C
            As[idx+1] = B
        As[0] /= np.linalg.norm(As[0].reshape(self.batch, -1),
                                axis=1)[:, None, None, None]
        return As
    
    def sweep(self, target_Bs):
        # calculate upper environments
        upper_envs = self._calculate_upper_environments(target_Bs)
        # prepare initial state
        B = np.zeros((self.batch, 1, self.d, 1),
                     dtype=np.complex128)
        B[:,:,0,:] = 1.
        Bs = [B.copy() for _ in range(self.L)]
        # sweep through layers (and upper environments) from bottom to top
        for layer, upper_env in enumerate(upper_envs):
            # first deal with right half of the system
            right_envs = self._calculate_right_environments(Bs, upper_env)
            left_env = self._calculate_left_environment(Bs, upper_env)
            # sweep through the gates from left to right
            gates = np.moveaxis(self.gates[:, self.Lc:, layer], 1, 0)
            updated_gates = np.zeros((self.batch, self.L-1-self.Lc,
                                      self.d**2, self.d**2),
                                     dtype=np.complex128)
            for i, (gate, right_env) in enumerate(zip(gates, right_envs)):
                idx = self.Lc + i
                # calculate environment and remove old two-qubit gate
                # we can reuse part of this calculation for updating left_env
                left_env = np.einsum('ijakb,ialc->ijlckb',
                                     left_env,
                                     upper_env[idx+1])
                left_env = np.einsum('ijklm,ilmanb->ijkanb',
                                     gate.conj()\
                                         .reshape(self.batch, *(self.d,)*4),
                                     left_env)
                left_env = np.einsum('ijkalb,ibmc->ijkalmc',
                                     left_env,
                                     Bs[idx+1])
                # only update gate if in self.gate_layout
                if self.gate_layout[idx, layer]:
                    environment = np.einsum('ijkalmb,iab->ijklm',
                                            left_env,
                                            right_env)
                    # update gate
                    U = self._update_gate(environment)
                    updated_gates[:, i] = U
                else:
                    updated_gates[:, i] = np.eye(self.d**2)
                # update left_env with new gate
                left_env = np.einsum('ijkalmb,ijnlm->ikanb',
                                     left_env,
                                     U.reshape(self.batch, *(self.d,)*4))
            # remove old gates from right half of upper_env
            upper_env = self.apply_one_layer_right_conj(layer, upper_env)
            # update old gates to new gates
            self.gates[:, self.Lc:, layer] = updated_gates
            # apply updated gates on the right half of MPS
            Bs = self.apply_one_layer_right(layer, Bs)
            # now deal with left half of the system
            left_envs = self._calculate_left_environments(Bs, upper_env)
            right_env = self._calculate_right_environment(Bs, upper_env)
            # sweep through the gates from right to left
            gates = np.moveaxis(self.gates[:, self.Lc-1::-1, layer], 1, 0)
            for i, (gate, left_env) in enumerate(zip(gates, left_envs)):
                idx = self.Lc-1 - i
                # calculate environment and remove old two-qubit gate
                # we can reuse part of this calculation for updating right_env
                right_env = np.einsum('iajb,ibkcl->iajkcl',
                                      upper_env[idx],
                                      right_env)
                right_env = np.einsum('ijklm,ialmbn->iajkbn',
                                      gate.conj()\
                                          .reshape(self.batch, *(self.d,)*4),
                                      right_env)
                right_env = np.einsum('iclb,iajkbm->iajkclm',
                                      Bs[idx],
                                      right_env)
                # only update gate if in self.gate_layout
                if self.gate_layout[idx, layer]:
                    environment = np.einsum('iab,iajkblm->ijklm',
                                            left_env,
                                            right_env)
                    # update gate
                    U = self._update_gate(environment)
                    self.gates[:, idx, layer] = U
                else:
                    self.gates[:, idx, layer] = np.eye(self.d**2)
                # update right_env with new gate
                right_env = np.einsum('iajkblm,inklm->iajbn',
                                      right_env,
                                      U.reshape(self.batch, *(self.d,)*4))
            # update MPS tensors with new gates in left half
            Bs = self.apply_one_layer_left(layer, Bs)
        return Bs
    
    def _overlap(self, Bs, target_Bs):
        overlap = np.ones((self.batch, 1, 1))
        for B, tB in zip(Bs, target_Bs):
            overlap = np.einsum('iab,iajc->ibjc', overlap, tB)
            overlap = np.einsum('ibja,ibjc->iac', overlap, B)
        return np.squeeze(overlap)
    
    def fully_contract(self):
        # prepare initial state
        B = np.zeros((self.batch, 1, self.d, 1))
        B[:,:,0,:] = 1.
        Bs = [B.copy() for _ in range(self.L)]
        # apply layers
        for layer in range(self.layers):
            Bs = self.apply_one_layer_right(layer, Bs)
            Bs = self.apply_one_layer_left(layer, Bs)
        return Bs
    
    def optimize_circuit(self, target_Bs, iters=20):
        target_Bs = [B.conj() for B in target_Bs]
        overlaps = [self._overlap(self.fully_contract(), target_Bs)]
        time = []
        # iterate
        for i in range(iters):
            start = perf_counter()
            Bs = self.sweep(target_Bs)
            time.append(perf_counter() - start)
            overlaps.append(self._overlap(Bs, target_Bs))
        return overlaps, np.asarray(time), Bs 

    def truncate(self, As, chi_max=2, which='right_to_left'):
        if which == 'right_to_left':
            # sweeps right to left
            # assumes left-canonical form, returns right-canonical form
            for i in range(len(As)-1, 0, -1):
                # two-site tensor
                A0 = As[i-1]
                A1 = As[i]
                theta = np.einsum('iajb,ibkc->iajkc', A0, A1)
                _, chil, _, _, chir = theta.shape
                theta = theta.reshape(self.batch, chil * self.d, self.d * chir)
                # singular value decomposition
                X, S, Y = np.linalg.svd(theta, full_matrices=False)
                # new bond dimension
                chi_new = min(S.shape[1],
                              max(self.chi_min, max(np.sum(S > self.svd_min, axis=1))),
                              chi_max)
                # normalize truncated state
                S = S[:, :chi_new]\
                    / np.sqrt(np.sum(S[:, :chi_new]**2, axis=1))[:, None]
                # truncate and save
                As[i-1] = (X[:, :, :chi_new] * S[:, None, :])\
                    .reshape(self.batch, -1, self.d, chi_new)
                As[i] = Y[:, :chi_new, :]\
                    .reshape(self.batch, chi_new, self.d, -1)
        elif which == 'left_to_right':
            # sweeps left to right
            # assumes right-canonical form, returns left-canonical form
            for i in range(len(As)-1):
                # two-site tensor
                A0 = As[i]
                A1 = As[i+1]
                theta = np.einsum('iajb,ibkc->iajkc', A0, A1)
                _, chil, _, _, chir = theta.shape
                theta = theta.reshape(self.batch, chil * self.d, self.d * chir)
                # singular value decomposition
                X, S, Y = np.linalg.svd(theta, full_matrices=False)
                # new bond dimension
                chi_new = min(S.shape[1],
                              max(self.chi_min, max(np.sum(S > self.svd_min, axis=1))),
                              chi_max)
                # normalize truncated state
                S = S[:, :chi_new]\
                    / np.sqrt(np.sum(S[:, :chi_new]**2, axis=1))[:, None]
                # truncate and save
                As[i] = X[:, :, :chi_new]\
                    .reshape(self.batch, -1, self.d, chi_new)
                As[i+1] = (S[:, :, None] * Y[:, :chi_new, :])\
                    .reshape(self.batch, chi_new, self.d, -1)
        else:
            raise ValueError("Wrong argument 'which' given for 'truncate'.")
        return As
    
    def add_one_layer(self, target_Bs):
        """
        Add one additional layer of gates and intializes them with a good
        guess such that the resulting state is closer to the target state.
        
        Also updates internal parameters such that 'optimize_circuit' can
        be called again.
        
        Parameters
        ----------
        target_Bs: list of L (batch, chi_l, 2, chi_r) ndarrays
            The MPS tensors of a batch of target states.
            (The same input as for 'optimize_circuit'.)
            
        """
        
        # contract conjugate of circuit to target state
        As = target_Bs.copy()
        As = self.left_canonical(As)
        # loop through the layers from top to bottom
        for layer in range(self.layers-1, -1, -1):
            # first half sweep from right to left
            gates = np.moveaxis(self.gates[:, :self.Lc:-1, layer], 1, 0)
            for i, gate in enumerate(gates):
                idx = self.L-1-i
                C, B = self.apply_gate(gate.transpose(0,2,1).conj(),
                                       As[idx-1],
                                       As[idx],
                                       which='CB')
                As[idx-1] = C
                As[idx] = B
            # second half sweep from left to right
            As = self.right_canonical(As)
            gates = np.moveaxis(self.gates[:, :self.Lc+1, layer], 1, 0)
            for i, gate in enumerate(gates):
                A, C = self.apply_gate(gate.transpose(0,2,1).conj(),
                                       As[i],
                                       As[i+1],
                                       which='AC')
                As[i] = A
                As[i+1] = C
            As = self.left_canonical(As)
        # truncate to bond dimension chi=d
        # could do this in one go, but truncating to intermediate
        # bond dimensions should give better results
        chi_max = max([A.shape[1] for A in As])
        while chi_max > self.d: # exponentially reduce chi until chi=d
            # set chi_max and truncate right to left, gives right-canonical MPS
            chi_max = max(self.d, int(chi_max/1.5))
            As = self.truncate(As, chi_max=chi_max, which='right_to_left')
            # set chi_max and truncate left to right, gives left-canonical MPS
            chi_max = max(self.d, int(chi_max/1.5))
            As = self.truncate(As, chi_max=chi_max, which='left_to_right')
        # A_0 ... A_Lc A_Lc+1 ... A_L-1 ---> A_0 ... A_Lc C B_Lc+1 ... B_L-1
        Lt = np.ones((self.batch, 1, 1))
        for i in range(self.L-1, self.Lc, -1):
            _, chil, d, chir = As[i].shape
            A = np.einsum('iajb,icb->iajc', As[i], Lt)\
                .reshape(self.batch, chil, d * chir)
            Qt, Lt = np.linalg.qr(A.transpose(0,2,1))
            As[i] = Qt.transpose(0,2,1).reshape(self.batch, chil, d, chir)
        C = Lt.transpose(0,2,1)
        # turn MPS into gate sequence
        sign = lambda x: 2 * (x >= 0) - 1
        new_gates = np.zeros((self.batch, self.L-1, d**2, d**2),
                             dtype=self.gates.dtype)
        U = np.zeros((self.batch, d**2, d, d),
                     dtype=self.gates.dtype)
        iso = np.einsum('iajb,ibkc->iajkc', As[0], As[1])\
                        .reshape(self.batch, d**2, d)
        U[:, :, 0, :] = iso
        U, R = np.linalg.qr(U.reshape(self.batch, d**2, d**2))
        U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
        new_gates[:, 0, :, :] = U
        for i in range(2, self.Lc+1):
            U = np.zeros((self.batch, d**2, d, d),
                         dtype=self.gates.dtype)
            iso = As[i].reshape(self.batch, d**2, d)
            U[:, :, 0, :] = iso
            U, R = np.linalg.qr(U.reshape(self.batch, d**2, d**2))
            U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
            new_gates[:, i-1, :, :] = U
        U = np.zeros((self.batch, d**2, d**2),
                     dtype=self.gates.dtype)
        iso = C.reshape(self.batch, d**2)
        U[:, :, 0] = iso
        U, R = np.linalg.qr(U)
        U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
        new_gates[:, self.Lc, :, :] = U
        for i in range(self.Lc+1, self.L-2):
            U = np.zeros((self.batch, d**2, d, d),
                         dtype=self.gates.dtype)
            iso = As[i].transpose(0,2,3,1)\
                                 .reshape(self.batch, d**2, d)
            U[:, :, 0, :] = iso
            U, R = np.linalg.qr(U.reshape(self.batch, d**2, d**2))
            U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
            U = U.reshape(self.batch, d**2, d, d)\
                 .transpose(0,1,3,2)\
                 .reshape(self.batch, d**2, d**2)
            new_gates[:, i, :, :] = U
        U = np.zeros((self.batch, d**2, d, d),
                     dtype=self.gates.dtype)
        iso = np.einsum('iajb,ibkc->ijkac', As[-2], As[-1])\
                        .reshape(self.batch, d**2, d)
        U[:, :, 0, :] = iso
        U, R = np.linalg.qr(U.reshape(self.batch, d**2, d**2))
        U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
        U = U.reshape(self.batch, d**2, d, d)\
             .transpose(0,1,3,2)\
             .reshape(self.batch, d**2, d**2)
        new_gates[:, -1, :, :] = U
        # update internal parameters
        self.layers += 1
        self.gates = np.append(new_gates[:, :, None, :, :], self.gates, axis=2)
        self.gate_layout = np.ones((self.L-1, self.layers), dtype=np.int32)
