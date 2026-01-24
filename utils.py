
import torch
import numpy as np
import scipy.sparse as sp
import numpy.ctypeslib as ctl
import os.path as osp
import random
from ctypes import c_int
from scipy.sparse import coo_matrix
import copy
import torch.nn.functional as F
import os



def sparseTensor_to_coomatrix(edge_idx, num_nodes):
    if edge_idx.shape == torch.Size([0]):
        adj = coo_matrix((num_nodes, num_nodes), dtype=np.int)
    else:
        row = edge_idx[0].cpu().numpy()
        col = edge_idx[1].cpu().numpy()
        data = np.ones(edge_idx.shape[1])
        adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.int)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    
def adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]

    ctl_lib = ctl.load_library("./models/csrc/libmatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )

    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

def cuda_csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    
    ctl_lib = ctl.load_library("./models/csrc/libcudamatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                         c_int]
    ctl_lib.FloatCSRMulDense.restypes = c_int

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    data_nnz = len(data)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDense(answer, data_nnz, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def project_conflicting(args, group_grads_flat):
    alpha = args.alpha
    new_grads = []
    num_groups = len(group_grads_flat)

    valid_indices = [k for k, g in enumerate(group_grads_flat) if g is not None]

    for i in range(num_groups):
        if group_grads_flat[i] is None:
            new_grads.append(None)
            continue

        g_i_curr = group_grads_flat[i]
        g_i_new = g_i_curr.clone()
        g_others_sum = torch.zeros_like(g_i_curr)

        similarities = []
        target_indices = []

        for j in valid_indices:
            if i == j: continue
            g_j_raw = group_grads_flat[j]
            norm_i = torch.norm(g_i_curr)
            norm_j = torch.norm(g_j_raw)

            if norm_i > 0 and norm_j > 0:
                cos_sim = torch.dot(g_i_curr, g_j_raw) / (norm_i * norm_j + 1e-8)
            else:
                cos_sim = torch.tensor(0.0, device=g_i_curr.device)

            similarities.append(cos_sim)
            target_indices.append(j)

        if not similarities:
            new_grads.append(g_i_new)
            continue
        sim_tensor = torch.stack(similarities)
        weights = F.softmax(sim_tensor, dim=0)

        for idx, j in enumerate(target_indices):
            g_j_raw = group_grads_flat[j]
            w_ij = weights[idx]

            dot_product = torch.dot(g_j_raw, g_i_curr)

            norm_sq_i = torch.dot(g_i_curr, g_i_curr)

            if dot_product < 0:

                if norm_sq_i > 1e-6:
                    projection = (dot_product / norm_sq_i) * g_i_curr
                    g_j_prime = g_j_raw - projection
                else:
                    g_j_prime = g_j_raw
                g_others_sum += w_ij * g_j_prime

            else:
                g_others_sum += w_ij * g_j_raw

        g_i_final = g_i_new + alpha * g_others_sum
        new_grads.append(g_i_final)

    return new_grads

def cal_psedoinverse(matrix):
    U, s, V = torch.svd(matrix)
    primary_sigma_indices = torch.where(s >= 1e-6)[0]
    s[primary_sigma_indices] = 1 / s[primary_sigma_indices]
    S = torch.diag(s)
    psedoinverse = V @ S @ U.T
    return psedoinverse

def get_nearest_oth_d(gr_locals, gu):
    import time
    start_time = time.time()

    A = gr_locals

    A_T = A.T
    c = gu

    AAT_1 = cal_psedoinverse(A @ A_T)

    Ac = A @ c.reshape(-1, 1)

    AAT_1_Ac = AAT_1 @ Ac

    d = c - (A_T @ AAT_1_Ac).reshape(-1)

    cal_time = time.time() - start_time
    return d, cal_time


def flatten_grads_vector(grad_dict):
    keys = sorted(grad_dict.keys())
    tensors = [grad_dict[k].view(-1) for k in keys]
    if not tensors:
        return None
    return torch.cat(tensors)


def unflatten_vector_to_dict(flat_grad, template_dict):
    new_grad_dict = {}
    keys = sorted(template_dict.keys())
    offset = 0

    for k in keys:
        param = template_dict[k]
        if torch.is_tensor(param) and param.dtype in [torch.float32, torch.float64]:
            numel = param.numel()

            if offset + numel > flat_grad.numel():
                raise RuntimeError(
                    f"Error unflattening {k}: needed {numel} elements, but flat_grad only has {flat_grad.numel() - offset} left.")

            new_grad_dict[k] = flat_grad[offset: offset + numel].view_as(param)
            offset += numel
        else:
            pass

    return new_grad_dict