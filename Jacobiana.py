import numpy as np
import scipy.sparse as sp


class Jacobiana():    
    def derivadas(self, Gs: sp.csr_matrix, Bs: sp.csr_matrix, Vr: np.ndarray, Vi: np.ndarray, Ir: np.ndarray, Ii: np.ndarray, idx_virtual: np.ndarray, idx_medido: np.ndarray) -> sp.csr_matrix:
        #Definir matrizes de suporte para montar as submatrizes da Jacobiana
        A = sp.diags(Vr) @ Gs + sp.diags(Vi) @ Bs
        C = sp.diags(Vi) @ Gs - sp.diags(Vr) @ Bs

        #Submatrizes das derivadas da potência ativa e reativa
        H_PVr = (A + sp.diags(Ir))[3:, 3:][idx_medido, :]
        H_PVi = (C + sp.diags(Ii))[3:, 3:][idx_medido, :]
        H_QVr = (C - sp.diags(Ii))[3:, 3:][idx_medido, :]
        H_QVi = (-A + sp.diags(Ir))[3:, 3:][idx_medido, :]

        Vm = np.sqrt(Vr**2 + Vi**2)

        H_VVr = sp.csr_matrix(sp.diags(Vr / Vm))[3:, 3:]
        H_VVi = sp.csr_matrix(sp.diags(Vi / Vm))[3:, 3:]

        H_IrVr = Gs[3:, 3:][idx_virtual, :]
        H_IrVi = -Bs[3:, 3:][idx_virtual, :]
        H_IiVr = Bs[3:, 3:][idx_virtual, :]
        H_IiVi = Gs[3:, 3:][idx_virtual, :]

        J_sparse = sp.bmat([
            [H_PVr, H_PVi], 
            [H_QVr, H_QVi],
            [H_VVr, H_VVi],
            [H_IrVr, H_IrVi],
            [H_IiVr, H_IiVi]
        ], format='csr')

        return J_sparse
