import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.sparse.linalg import splu
from scipy.sparse import csr_matrix

import time

from dss import DSS as dss_engine
from Jacobiana import Jacobiana
from Residuos import Residuo

class EESD():
    def __init__(self, master_path, baseva: float = 1e8, verbose: bool = False) -> None:
        self.baseva = baseva
        self.MasterFile = master_path
        self.verbose = verbose
        self.DSSCircuit, self.DSSText, self.DSSObj, self.DSSMonitors = self.InitializeDSS()
        
        self.resolve_fluxo_carga()
        if self.verbose: print('Acabou o fluxo de carga')
        
        self.barras, self.num_medidas = self.medidas(self.baseva)

        Ybus, node_order = self.get_Ybus(False)
        Ybus = Ybus.tolil()

        self.Ybus, self.nodes = self.organiza_Ybus(Ybus)
        self.Ybus = self.Conserta_Ybus(self.Ybus)
        self.Gs = csr_matrix(np.real(self.Ybus).toarray())
        self.Bs = csr_matrix(np.imag(self.Ybus).toarray())

        self.vet_estados = self.iniciar_vet_estados()
        
        if self.verbose: print('Matriz de adimitância modificada com sucesso')

        self.matriz_pesos, self.dp = self.Calcula_pesos()

        self.residuos = Residuo(self.barras)
        self.jac = Jacobiana()

    def resolve_fluxo_carga(self):
        self.DSSText.Command = 'Clear'
        self.DSSText.Command = f'Compile {self.MasterFile}'

        self.iniciar_medidores()

        self.DSSText.Command = 'Set ControlMode=OFF'

        self.DSSText.Command = 'Solve'
        if not self.DSSCircuit.Solution.Converged:
            raise RuntimeError('O fluxo de carga não convergiu')

    def InitializeDSS(self):
        DSSObj = dss_engine
        flag = DSSObj.Start(0)
        if flag and self.verbose:
            print('OpenDSS COM Interface initialized succesfully.')
            
        elif self.verbose:
            print('OpenDSS COMInterface failed to start.')
            
        #Set up the interface variables - Comunication OpenDSS with Python 
        DSSText = DSSObj.Text
        DSSCircuit = DSSObj.ActiveCircuit
        DSSMonitors = DSSCircuit.Monitors

        return DSSCircuit, DSSText, DSSObj, DSSMonitors

    def reset(self):
        self.resolve_fluxo_carga()
        self.barras, self.num_medidas = self.medidas(self.baseva)
        self.vet_estados = self.iniciar_vet_estados()
    
    def iniciar_medidores(self) -> None:
        count = 0
        for i, barra in enumerate(self.DSSCircuit.AllBusNames):
            self.DSSCircuit.SetActiveBus(barra)
            #Não incluir a barra slack
            pce_elements = [elem.split('.')[0] for elem in self.DSSCircuit.ActiveBus.AllPCEatBus]
            if 'Vsource' in pce_elements:
                continue

            for elem in self.DSSCircuit.Buses.AllPCEatBus:
                if 'Load' in elem or 'Generator' in elem or 'PVSystem' in elem:
                    self.DSSText.Command = f'New Monitor.pqi{count} element={elem}, terminal=1, mode=1, ppolar=no'
                    count += 1
            
            #Escolher elemento com maior numero de fases da barra
            max_fases = 0
            elem = 'None'
            for pde in self.DSSCircuit.Buses.AllPDEatBus:
                self.DSSCircuit.SetActiveElement(pde)
                num_fases = len(self.DSSCircuit.ActiveCktElement.NodeOrder)
                if num_fases > max_fases:
                    elem = self.DSSCircuit.ActiveCktElement.Name
                    max_fases = num_fases
            #Colocar um medidor no elemento escolhido
            if elem != 'None':
                self.DSSCircuit.SetActiveElement(elem)
                if self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0] == barra:
                    self.DSSText.Command = f'New Monitor.v{i} element={elem}, terminal=1, mode=32'
                    
                elif self.DSSCircuit.ActiveCktElement.BusNames[1].split('.')[0] == barra:
                    self.DSSText.Command = f'New Monitor.v{i} element={elem}, terminal=2, mode=32'
                    
                else:
                    print(f'Nenhum elemento conectado na barra {barra}')

    def indexar_barras(self) -> pd.DataFrame:
        nomes = []
        bases = []
        for barra in self.DSSCircuit.AllBusNames:

            self.DSSCircuit.SetActiveBus(barra)

            #Não incluir a barra slack
            pce_elements = [elem.split('.')[0] for elem in self.DSSCircuit.ActiveBus.AllPCEatBus]
            if 'Vsource' in pce_elements:
                self.slack = self.DSSCircuit.ActiveBus.Name
                continue

            #Base é em fase-neutro
            base = self.DSSCircuit.Buses.kVBase
            
            if base == 0:
                raise ValueError('Tensão base não pode ser 0')
            
            nomes.append(barra)
            bases.append(base)

        idx = [i for i in range(len(nomes))]
        
        barras = pd.DataFrame(columns=['Nome', 'Base', 'Fases', 'Inj_pot_at', 'Inj_pot_rat', 'Tensao'],
                            index=idx)

        barras['Nome'] = nomes
        barras.loc[idx, 'Base'] = bases

        return barras

    def gera_medida_imperfeita(self, medida: float) -> None:
        # Gerar fatores aleatórios com base na distribuição normal
        fatores = np.random.normal(medida, self.dp, self.num_medidas)
        
        for i, medidas in enumerate(self.barras['Inj_pot_at']):
            self.barras['Inj_pot_at'][i] = medidas + medidas * fatores[i*3:(i+1)*3]

    def iniciar_vet_estados(self) -> np.array:
        Vr = np.zeros(len(self.fases))
        Vi = np.zeros(len(self.fases))
        angulos = [0, -2*np.pi/3, 2*np.pi/3]
        
        for i, fase in enumerate(self.fases):
            if fase == 3:
                Vr[i] = 1e-5
                Vi[i] = 1e-5
                continue
            Vr[i] = np.cos(angulos[fase])
            Vi[i] = np.sin(angulos[fase])
        
        vet_estados = np.concatenate((Vr, Vi))
                
        return vet_estados
    
    def achar_index_barra(self, barras: pd.DataFrame, barra: int) -> int:
        #Retorna o index da barra do monitor ativo
        self.DSSCircuit.SetActiveElement(self.DSSMonitors.Element)
        
        self.DSSCircuit.SetActiveBus(self.DSSCircuit.ActiveCktElement.BusNames[barra])
        nome = self.DSSCircuit.Buses.Name
        
        return barras.index[barras['Nome'] == nome].to_list()[0]

    def pegar_fases(self, fases: list) -> np.array:
        fases = list(dict.fromkeys(fases))
        fases = [fase for fase in fases if fase != -1]
        
        return fases

    def medidas(self, baseva: int) -> pd.DataFrame:
        #Verivicar se os indexes podem ser substituídos pelos nomes das colunas sem perda de performance
        barras = self.indexar_barras()
        
        num_medidas = 0
        for idx, bus in enumerate(barras['Nome']):
            self.DSSCircuit.SetActiveBus(bus)
            fases = self.DSSCircuit.ActiveBus.Nodes - 1
            fases = self.pegar_fases(fases)
            #Preenche a coluna das fases
            barras.iat[idx, 2] = fases

            #Inicializa as medidas como 0 para injeção ativa e reativa
            barras.iat[idx, 3] = np.array([0, 0, 0, 0], dtype=np.float64)
            barras.iat[idx, 4] = np.array([0, 0, 0, 0], dtype=np.float64)
            num_medidas += len(fases)*2
        
        #Amostra e salva os valores dos medidores do sistema
        self.DSSMonitors.ResetAll()
        self.DSSMonitors.SampleAll()
        self.DSSMonitors.SaveAll()
        
        self.DSSMonitors.First
        for _ in range(self.DSSMonitors.Count):
            barra = self.DSSMonitors.Terminal - 1
            index_barra = self.achar_index_barra(barras, barra)

            #Pegar as fases da carga atual
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            fases = self.pegar_fases(fases)

            matriz_medidas = self.DSSMonitors.AsMatrix()[0][2:]

            if 'pqi' in self.DSSMonitors.Name:
                medidas_at = np.zeros(4)
                medidas_rat = np.zeros(4)
                
                for i, fase in enumerate(fases):
                    medidas_at[fase] = matriz_medidas[i*2]
                    medidas_rat[fase] = matriz_medidas[i*2+1]

                #Atualiza as medidas de injeção ativa e reativa
                barras.iat[index_barra, 3] += -medidas_at*1000 / baseva
                barras.iat[index_barra, 4] += -medidas_rat*1000 / baseva
                
            elif 'v' in self.DSSMonitors.Name:
                if type(barras['Tensao'][index_barra]) != np.ndarray:
                    medidas = np.zeros(4)
                    self.DSSCircuit.SetActiveBus(barras['Nome'][index_barra])
                    fases = self.DSSCircuit.ActiveBus.Nodes - 1
                    fases = self.pegar_fases(fases)
                    
                    matriz_medidas = self.DSSCircuit.Buses.puVmagAngle[::2]

                    #Atualiza as medidas de tensão 
                    for i, fase in enumerate(fases):
                        medidas[fase] = matriz_medidas[i]

                    barras.iat[index_barra, 5] = medidas
                    num_medidas += len(fases)
            
            self.DSSMonitors.Next
            
        return barras, num_medidas

    def forma_matriz(self, fases: np.ndarray, fases_barra: list, Yprim: list) -> np.array:
        temp = np.zeros((len(fases_barra), len(fases_barra)), dtype=np.complex128)
        fases = [fase for fase in fases if fase != -1]
        fases_barra = list(fases_barra-1)
        idx = []
        for fase in fases:
            idx.append(fases_barra.index(fase))
            
        k = 0
        for i in idx:
            for j in idx:
                temp[i, j] = Yprim[k]
                k += 1

        if len(Yprim) == 16:
            temp = np.reshape(Yprim, (4, 4))
        
        Yprim = temp
        
        return Yprim
    
    def _get_node_order(self):
        return list(self.DSSCircuit.YNodeOrder)
 
    def _get_system_y(self):
        n = self.DSSCircuit.NumNodes
        data, indices, indptr = self.DSSObj.YMatrix.GetCompressedYMatrix(factor=False)
        return sp.csc_matrix((data, indices, indptr), shape=(n, n))

    def get_Ybus(self, return_with_loads=False):
        node_order = self._get_node_order()
    
        Y_com_carga = self._get_system_y() if return_with_loads else None
    
        load_names = []
        idx = self.DSSCircuit.Loads.First
        while idx:
            load_names.append(self.DSSCircuit.Loads.Name)
            idx = self.DSSCircuit.Loads.Next
    
        original_enabled = {}
        for name in load_names:
            self.DSSCircuit.SetActiveElement(f"Load.{name}")
            original_enabled[name] = self.DSSCircuit.ActiveCktElement.Enabled
    
        for name in load_names:
            self.DSSText.Command = f"Load.{name}.enabled=No"
        self.DSSText.Command = "solve"
    
        node_order_sem_carga = self._get_node_order()
        if node_order_sem_carga != node_order:
            raise RuntimeError("A ordem/numero de nos mudou ao desabilitar as cargas. "
                               "Verifique se alguma carga estava sustentando um no que fica "
                               "isolado (ilha) do resto da rede quando ela e desligada."
                               )
    
        Y_sem_carga = self._get_system_y()
    
        # --- restaura o estado original das cargas ---
        for name in load_names:
            estado = "Yes" if original_enabled[name] else "No"
            self.DSSText.Command = f"Load.{name}.enabled={estado}"
        self.DSSText.Command = "solve"
    
        if return_with_loads:
            extras = {
                "Y_com_carga": Y_com_carga,
                "Y_carga": Y_com_carga - Y_sem_carga,
            }
            return Y_sem_carga, node_order, extras
    
        return Y_sem_carga, node_order

    def organiza_Ybus(self, Ybus):
        idx_original = {
            node.lower(): i for i, node in enumerate(self.DSSCircuit.YNodeOrder)
        }

        barras_ordenadas = []
        for barra, fases in zip(self.barras['Nome'], self.barras['Fases']):
            for fase in fases:
                barras_ordenadas.append(f'{barra}.{fase+1}')

        self.DSSCircuit.SetActiveBus(self.slack)
        slack = [f'{self.slack}.{fase}' for fase in self.DSSCircuit.ActiveBus.Nodes]
        nova_ordem = slack + barras_ordenadas

        perm = [idx_original[bus.lower()] for bus in nova_ordem]

        Ybus = Ybus.tocsr()[perm, :].tocsc()[:, perm].tocsr()

        idx_new = [i for i, _ in enumerate(nova_ordem)]

        all_node_names = nova_ordem
        nodes = {
            all_node_names[orig_idx].lower(): new_idx
            for new_idx, orig_idx in enumerate(idx_new)
        }

        assert len(nodes) == Ybus.shape[0], (
            "nodes e Ybus com tamanhos diferentes -- nao deveria acontecer "
            "com essa construcao, indica bug na propria funcao"
        )
        
        fases = []
        for node in nodes:
            fases.append(int(node.split('.')[1])-1)

        self.fases = fases[3:]

        return Ybus, nodes
    
    def Conserta_Ybus(self, Ybus):
        #TODO: Generalizar para nomes diferentes
        self.DSSCircuit.SetActiveElement('Vsource.source')
        Yprim = self.DSSCircuit.ActiveCktElement.Yprim
        real = Yprim[::2]
        imag = Yprim[1::2]*1j
        Yprim = real+imag
        Yprim = np.reshape(Yprim, (6, 6))
        Ybus[:3, :3] -= Yprim[:3, :3]

        self.DSSCircuit.SetActiveBus(self.slack)
        base = self.DSSCircuit.Buses.kVBase
        v_base = [base for _ in range(self.DSSCircuit.Buses.NumNodes)]

        for fases, base in zip(self.barras['Fases'], self.barras['Base']):
            v_base += [base for _ in fases]

        V_diag = sp.diags(v_base) * 1000
        Ybus = (V_diag @ Ybus @ V_diag) / self.baseva

        return Ybus

    def Calcula_pesos(self) -> tuple:
        inj_pot_at = []
        inj_pot_rat = []
        tensoes = []
        for fases, pot_at, pot_rat, tensao in zip(self.barras['Fases'], self.barras['Inj_pot_at'], self.barras['Inj_pot_rat'], self.barras['Tensao']):
            for fase in fases:
                inj_pot_at.append(pot_at[fase])
                inj_pot_rat.append(pot_rat[fase])
                tensoes.append(tensao[fase])

        inj_pot_at = np.array(inj_pot_at)
        inj_pot_rat = np.array(inj_pot_rat)

        idx_virtual = np.where(inj_pot_at == 0)[0]
        idx_medido = np.where(inj_pot_at != 0)[0]

        Ir = np.zeros(len(idx_virtual))
        Ii = np.zeros(len(idx_virtual))
        medidas = np.concatenate([inj_pot_at[idx_medido], inj_pot_rat[idx_medido], tensoes, Ir, Ii])

        dp = (medidas * 0.01) / (3 * 100)
        dp[dp == 0] = 1e-5
        pesos = dp**-2
        pesos[pesos > 1e10] = 1e10
        matriz_pesos = np.diag(pesos)
        
        return sp.csr_matrix(matriz_pesos), np.abs(dp)
    
    def run(self, max_error: float, max_iter: int) -> np.array:       
        W = self.matriz_pesos
        k = 0
        self.residuo = 1
        delx = 1
        while(np.max(np.abs(delx)) > max_error and max_iter > k):
            inicio = time.time()
            self.DSSCircuit.SetActiveBus(self.slack)
            Vr_ref = self.DSSCircuit.Buses.puVoltages[::2]
            Vi_ref = self.DSSCircuit.Buses.puVoltages[1::2]
    
            Vr = np.concatenate([Vr_ref, self.vet_estados[:len(self.vet_estados)//2]])
            Vi = np.concatenate([Vi_ref, self.vet_estados[len(self.vet_estados)//2:]])

            self.residuo, Ir, Ii = self.residuos.calc_res(self.Gs, self.Bs, Vr, Vi)

            idx_virtual = self.residuos.idx_virtual
            idx_medido = self.residuos.idx_medido

            J = self.jac.derivadas(self.Gs, self.Bs, Vr, Vi, Ir, Ii, idx_virtual, idx_medido)

            fim_res = time.time()
            idx = np.where(np.max(np.abs(self.residuo)) == np.abs(self.residuo))
            
            fim_jac = time.time()
            
            Jt = J.T.tocsr()
            
            JtW = Jt.multiply(W.diagonal())
            
            G = JtW @ J

            b = JtW @ self.residuo

            solver = splu(G.tocsc())
            delx = solver.solve(b)

            self.vet_estados += delx
            fim = time.time()
            
            k += 1
            if np.isnan(self.vet_estados.any()):
                print('NaN na solução')
                break
            if k == 99:
                print('Não convergiu')
            if self.verbose:
                print(f'Os resíduos da iteração {k} levaram {fim_res-inicio:.3f}s')
                print(f'A jacobiana da iteração {k} levou {fim_jac-fim_res:.3f}s')
                print(f'Atualizar o vetor de estados da iteração {k} levou {fim-fim_jac:.3f}')
                print(f'A iteração {k} levou {fim-inicio:.3f}s')

        return self.vet_estados