import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class Residuo():
    def __init__(self, barras: pd.DataFrame) -> None:
        self.barras = barras
        self._precomputa_refs()
        
    def _precomputa_refs(self) -> None:
        inj_at, inj_rat, tensoes = [], [], []

        for fases, pot_at, pot_rat, tensao in zip(
            self.barras['Fases'],
            self.barras['Inj_pot_at'],
            self.barras['Inj_pot_rat'],
            self.barras['Tensao']
        ):
            for fase in fases:
                inj_at.append(pot_at[fase])
                inj_rat.append(pot_rat[fase])
                tensoes.append(tensao[fase])

        self._inj_at_ref  = np.array(inj_at)
        self._inj_rat_ref = np.array(inj_rat)
        self._tensao_ref  = np.array(tensoes)
        self.idx_virtual = np.where(self._inj_at_ref == 0)[0]
        self.idx_medido = np.where(self._inj_at_ref != 0)[0]

    def calc_res(self, Gs: csr_matrix, Bs: csr_matrix, Vr: np.ndarray, Vi: np.ndarray) -> np.ndarray:
        Ir = Gs @ Vr - Bs @ Vi
        Ii = Gs @ Vi + Bs @ Vr

        P = Vr * Ir + Vi * Ii
        Q = Vi * Ir - Vr * Ii

        V = np.sqrt(Vr**2 + Vi**2)

        res_inj_pot_at = self._inj_at_ref[self.idx_medido] - P[3:][self.idx_medido]
        res_inj_pot_rat = self._inj_rat_ref[self.idx_medido] - Q[3:][self.idx_medido]
        res_tensao = self._tensao_ref  - V[3:]
        res_corr_real = np.zeros(len(self.idx_virtual)) - Ir[3:][self.idx_virtual]
        res_corr_imag = np.zeros(len(self.idx_virtual)) - Ii[3:][self.idx_virtual]

        return np.concatenate([res_inj_pot_at, res_inj_pot_rat, res_tensao, res_corr_real, res_corr_imag]), Ir, Ii
