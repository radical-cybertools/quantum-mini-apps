import numpy as np

def FRQI_RGBa_encoding(images, indexing=None):
    """
    images : (batch, m, n, 3) ndarray
    
    indexing : None, 'hierarchical' or 'snake'
    
    """
    # images
    batch, m, n, _ = images.shape
    # apply indexing
    if indexing == 'hierarchical':
        num_m = int(np.log2(m))
        num_n = int(np.log2(n))
        images = images.reshape(batch, *(2,)*(num_m + num_n), 3)
        images = images.transpose(0,
                                  *[ax+1 for bit in range(min(num_m, num_n)) for ax in [bit, bit + num_m]],
                                  *range(min(num_m, num_n) + 1, num_m + 1),
                                  *range(min(num_m, num_n) + num_m + 1, num_m + num_n + 1),
                                  -1)
    elif indexing == 'snake':
        images[:, ::2, :] = images[:, ::2, ::-1]
    images = images.reshape(batch, m*n, 3)
    # map pixels to states
    states = np.zeros((batch, 2**3, m*n))
    funcs = [lambda x: np.cos(np.pi * x/2), lambda x: np.sin(np.pi * x/2)]
    for i in range(3):
        states[:, i  , :] = funcs[0](images[:, :, i])
        states[:, i+4, :] = funcs[1](images[:, :, i])
    states[:, 3, :] = 1.
    # normalize states
    states = states.reshape(batch, 2**3*m*n)/np.sqrt(m*n)/2
    return states[0]


def FRQI_RGBa_decoding(states, indexing=None, shape=(32,32)):
    """
    states : (batch, 2**3 * m * n) ndarray
    
    indexing : None, 'hierarchical' or 'snake'
    
    shape : tuple (m, n)
    
    """
    # states
    if len(states.shape) > 1:
        batch = states.shape[0]
    else:
        batch = 1
    # batch = states.shape[0]
    # invert indexing
    if indexing == 'hierarchical':
        num_m = int(np.log2(shape[0]))
        num_n = int(np.log2(shape[1]))
        states = states.reshape(batch * 2**3, *(2,)*(num_m+num_n))
        if num_m > num_n:
            states = states.transpose(0, *range(1, 2*num_n+1, 2), *range(2*num_n+1, num_m+num_n), *range(2, 2*num_n+1, 2))
        else:
            states = states.transpose(0, *range(1, 2*num_m+1, 2), *range(2, 2*num_m+1, 2), *range(2*num_m+1, num_m+num_n))
    elif indexing == 'snake':
        states = states.reshape(batch * 2**3, *shape)
        states[:, ::2, :] = states[:, ::2, ::-1]
    # map states to pixels
    channels = []
    states = states.reshape(batch, 2**3, -1)
    for i in range(3):
        channel = (states[:, i, :]**2 - states[:, i+4, :]**2) * states.shape[-1] * 4
        channel[channel > 1.] = 1.
        channel[channel <-1.] =-1.
        channels.append(np.arccos(channel)/np.pi)
    images = np.stack(channels, axis=-1).reshape(batch, *shape, 3)
    return images[0]

def calc_MPS(states, chi_max=256, normalize=False, d=2):
    """
    Calculate MPS tensors from d^L state amplitudes.
    
    """
    
    if len(states.shape) > 1:
        batchsize = states.shape[0]
    else:
        batchsize = 1
    states = states.reshape(batchsize, -1)
    L = states.shape[-1]
    L = int(np.log(L)/np.log(d))
    
    A_tensors = []
    Lambda_tensors = []
    
    psi_rest = states.reshape(batchsize, 1, -1)
    
    for bond in range(1, L):
        # bond dimensions of previous step
        _, chi, dim_rest = psi_rest.shape
        assert dim_rest == d**(L-bond+1)
        
        # move cut by one site and reshape wave function
        psi_rest = psi_rest.reshape(batchsize, chi*d, dim_rest//d)
        
        # perform SVD
        A, Lambda, psi_rest = np.linalg.svd(psi_rest, full_matrices=False)
        
        # and truncate
        chi_new = min(A.shape[-1], chi_max)
        
        A_tensors.append(A[:,:,:chi_new].reshape(batchsize, chi, d, chi_new))
        Lambda = Lambda[:,:chi_new]
        if normalize:
            Lambda /= np.sqrt(np.sum(Lambda**2, axis=1))[:,None]
        Lambda_tensors.append(Lambda)
        psi_rest = psi_rest[:,:chi_new,:]
        
        # multiply Schmidt values to wave function
        psi_rest = Lambda[:,:,None] * psi_rest
    
    # save last MPS tensor
    A_tensors.append(psi_rest.reshape(batchsize, chi_new, d, 1))
    
    return A_tensors, Lambda_tensors

def calc_state(A_list, renormalize=True):
    """
    Calculate full state, i.e., all 2^L state amplitudes, from MPS.
    
    """
    
    states = A_list[0]
    batchsize = states.shape[0]
    for A in A_list[1:]:
        states = np.einsum(states, [*np.arange(len(states.shape))],#         0, 1, ..., l-1
                           A, [0, *(np.arange(3) + len(states.shape)) - 1],# 0, l-1, l, l+1
                           [*np.arange(len(states.shape)-1), *(np.arange(2)+len(states.shape))])
    states = states.reshape(batchsize, -1)
    if renormalize:
        states /= np.sqrt(np.einsum('ij,ij->i', states.conj(), states))[:,None]
    return states


def right_canonical(As):
    batchsize = As[0].shape[0]
    Bs = []
    Lt = np.ones((batchsize, 1, 1))
    for A in As[::-1]:
        _, chil, d, chir = A.shape
        A = np.einsum('iajb,icb->iajc', A, Lt)\
            .reshape(batchsize, chil, d * chir)
        Qt, Lt = np.linalg.qr(A.transpose(0,2,1))
        Bs.append(Qt.transpose(0,2,1).reshape(batchsize, chil, d, chir))
    return Bs[::-1]