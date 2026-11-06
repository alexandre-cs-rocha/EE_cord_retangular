import time

import numpy as np
import pandas as pd

import scipy.sparse as sp
from scipy.sparse.linalg import splu

from dss import DSS as dss_engine
from Jacobiana import Jacobiana
from Residuos import Residuo

class EESD():
    def __init__(self, master_path, baseva: float = 1e8, verbose: bool = False, inicializar_neutros_direto: bool = False, inicializar_completo_direto: bool = False) -> None:
        self.baseva = baseva
        self.MasterFile = master_path
        self.verbose = verbose
        self.inicializar_neutros_direto = inicializar_neutros_direto
        self.inicializar_completo_direto = inicializar_completo_direto
        self.DSSCircuit, self.DSSText, self.DSSObj, self.DSSMonitors = self.InitializeDSS()
        
        self.resolve_fluxo_carga()
        if self.verbose: print('Acabou o fluxo de carga')

        self.barras, self.num_medidas = self.medidas(self.baseva)
        
        Ybus, node_order = self.get_Ybus(False)
        Ybus = Ybus.tocsr()

        self.Ybus, self.nodes = self.organiza_Ybus(Ybus)
        self.Ybus = self.Conserta_Ybus(self.Ybus)

        nos_isolados = self.identifica_nos_isolados()

        # Ordem original da Ybus
        ordem_ybus = [f"{self.slack}.{fase}".lower() for fase in (1, 2, 3)]

        ordem_ybus += [
            f"{barra}.{int(fase) + 1}".lower()
            for barra, fases in zip(
                self.barras["Nome"],
                self.barras["Fases"]
            )
            for fase in fases
        ]


        def chave_no(nome):
            barra, no = nome.rsplit(".", 1)
            return barra.lower(), int(no)


        assert len(ordem_ybus) == self.Ybus.shape[0], (
            "A ordem construída não corresponde à Ybus."
        )

        # Índices que serão mantidos
        idx_keep = [
            i for i, no in enumerate(ordem_ybus)
            if chave_no(no) not in nos_isolados
        ]

        linhas_keep = []
        fases_keep = []

        for i, (barra, fases) in enumerate(
            zip(self.barras["Nome"], self.barras["Fases"])
        ):
            fases_validas = [
                fase for fase in fases
                if (str(barra).lower(), int(fase) + 1)
                not in nos_isolados
            ]

            if fases_validas:
                linhas_keep.append(i)
                fases_keep.append(fases_validas)

        self.barras = self.barras.iloc[linhas_keep].copy()
        self.barras["Fases"] = fases_keep
        self.barras.reset_index(drop=True, inplace=True)

        # Filtra linhas e colunas da Ybus
        self.Ybus = self.Ybus[idx_keep, :][:, idx_keep]

        ordem_ybus_filtrada = [ordem_ybus[i] for i in idx_keep]

        if ordem_ybus_filtrada[:3] != [
            f"{self.slack}.{fase}".lower() for fase in (1, 2, 3)
        ]:
            raise RuntimeError(
                "A filtragem alterou/removeu nós da barra slack. "
                "O estimador pressupõe três nós slack nas posições 0, 1 e 2."
            )

        # Reindexação compacta: 0, 1, ..., N_filtrado-1.
        self.nodes = {
            nome_no: idx
            for idx, nome_no in enumerate(ordem_ybus_filtrada)
        }
        self.node_order = ordem_ybus_filtrada

        self.fases = np.asarray(
            [int(nome_no.rsplit('.', 1)[1]) - 1
                for nome_no in ordem_ybus_filtrada[3:]],
            dtype=int
        )
        self.idx_keep = idx_keep

        self.Gs = self.Ybus.real
        self.Bs = self.Ybus.imag

        self.vet_estados = self.iniciar_vet_estados_sem_carga()
        if self.inicializar_completo_direto:
            self.vet_estados = self.inicializar_fluxo_direto_completo(
                self.vet_estados
            )
        elif self.inicializar_neutros_direto:
            self.vet_estados = self.inicializar_neutros_fluxo_direto(
                self.vet_estados
            )
        
        if self.verbose: print('Matriz de adimitância modificada com sucesso')

        self.matriz_pesos, self.dp = self.Calcula_pesos(fator_sigma_neutro=0.3)

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

    def iniciar_vet_estados_sem_carga(self):

        # Recuperar tensão dos nós slack
        self.DSSCircuit.SetActiveBus(self.slack)

        Vr_ref = np.asarray(
            self.DSSCircuit.Buses.puVoltages[::2]
        )

        Vi_ref = np.asarray(
            self.DSSCircuit.Buses.puVoltages[1::2]
        )

        Vslack = Vr_ref + 1j * Vi_ref

        if len(Vslack) != 3:
            raise RuntimeError(
                "A inicialização pressupõe três nós slack"
            )

        # Particionar a Ybus
        Y = self.Ybus.tocsc()

        Yuu = Y[3:, 3:].tocsc()
        Yus = Y[3:, :3].tocsc()

        # Resolver a rede passiva sem cargas
        try:
            lu = splu(Yuu)
            V0 = lu.solve(-(Yus @ Vslack))

        except RuntimeError as exc:
            raise RuntimeError(
                "Não foi possível calcular a inicialização "
                "sem carga. Verifique se Yuu é singular."
            ) from exc

        if not np.all(np.isfinite(V0)):
            raise FloatingPointError(
                "A inicialização produziu valores inválidos"
            )

        # Evitar fasores exatamente nulos nos neutros
        for i, fase in enumerate(self.fases):

            if fase == 3 and abs(V0[i]) < 1e-8:
                V0[i] = 1e-5 + 1j*1e-5

        return np.concatenate((V0.real, V0.imag))

    def extrair_tensoes_pu_por_no(self):
        tensoes = {}

        for barra in self.DSSCircuit.AllBusNames:

            self.DSSCircuit.SetActiveBus(barra)

            nos = np.asarray(
                self.DSSCircuit.ActiveBus.Nodes,
                dtype=int
            ).ravel()

            v_pu = np.asarray(
                self.DSSCircuit.Buses.puVoltages,
                dtype=float
            ).ravel()

            if v_pu.size != 2 * nos.size:
                raise RuntimeError(
                    f"Quantidade inconsistente de tensões na barra {barra}: "
                    f"{v_pu.size} valores para {nos.size} nós."
                )

            for k, no in enumerate(nos):

                nome_no = f"{barra}.{no}".lower()

                tensoes[nome_no] = (
                    v_pu[2*k] + 1j*v_pu[2*k + 1]
                )

        return tensoes

    def _resolver_tensoes_fluxo_direto(self):
        sol = self.DSSCircuit.Solution
        load_model_original = sol.LoadModel

        try:
            # dssAdmittance: cargas representadas por admitâncias equivalentes.
            sol.LoadModel = 2
            sol.SolveDirect()

            # Copiar as tensões antes de restaurar o modelo original.
            tensoes_diretas = self.extrair_tensoes_pu_por_no()

        finally:
            # Restaurar a configuração e a solução normal do circuito.
            sol.LoadModel = load_model_original
            sol.Solve()

            if not sol.Converged:
                raise RuntimeError(
                    "O fluxo de carga normal não convergiu após restaurar "
                    "o LoadModel original."
                )

            self.DSSCircuit.SetActiveBus(self.slack)

        return tensoes_diretas


    def _aplicar_tensoes_fluxo_direto(self, vet_estados, *, somente_neutros: bool):
        vet_estados = np.asarray(vet_estados, dtype=float).copy()
        n_estados = len(self.fases)

        if vet_estados.size != 2 * n_estados:
            raise RuntimeError(
                "Dimensão incompatível entre vet_estados e self.fases: "
                f"{vet_estados.size} valores para {n_estados} estados por parte."
            )

        tensoes_diretas = self._resolver_tensoes_fluxo_direto()
        tensoes_aplicadas = {}
        indices_atualizados = set()

        for nome_no, idx_ybus in self.nodes.items():
            # Os três primeiros nós são os nós de referência (slack).
            if idx_ybus < 3:
                continue

            idx_estado = idx_ybus - 3

            if not 0 <= idx_estado < n_estados:
                raise RuntimeError(
                    "Índice de nó incompatível com o vetor de estados: "
                    f"nó={nome_no}, índice Ybus={idx_ybus}, "
                    f"índice de estado={idx_estado}, "
                    f"quantidade de estados por parte={n_estados}. "
                    "Verifique a reindexação de self.nodes após filtrar a Ybus."
                )

            # Na convenção atual, fase 3 (índice 3) representa o neutro .4.
            eh_neutro = self.fases[idx_estado] == 3
            if somente_neutros and not eh_neutro:
                continue

            chave_no = str(nome_no).lower()
            if chave_no not in tensoes_diretas:
                raise KeyError(
                    f"Nó {chave_no} não encontrado nas tensões "
                    "retornadas pelo OpenDSS."
                )

            v = tensoes_diretas[chave_no]
            if not np.isfinite(v.real) or not np.isfinite(v.imag):
                raise RuntimeError(
                    f"Tensão inválida no nó {chave_no}: {v}"
                )

            # O vetor de estados é [Vr dos nós não-slack, Vi dos nós não-slack].
            vet_estados[idx_estado] = v.real
            vet_estados[n_estados + idx_estado] = v.imag

            tensoes_aplicadas[chave_no] = v
            indices_atualizados.add(idx_estado)

        indices_esperados = {
            i for i, fase in enumerate(self.fases)
            if (not somente_neutros) or fase == 3
        }

        faltantes = sorted(indices_esperados - indices_atualizados)
        if faltantes:
            exemplos = [
                str(self.node_order[i + 3])
                if hasattr(self, "node_order") and i + 3 < len(self.node_order)
                else str(i)
                for i in faltantes[:10]
            ]
            tipo = "neutros" if somente_neutros else "nós"
            raise RuntimeError(
                f"A solução direta não forneceu {tipo} para todos os estados. "
                f"Faltam {len(faltantes)} índices; exemplos: {exemplos}."
            )

        return vet_estados, tensoes_aplicadas


    def inicializar_neutros_fluxo_direto(self, vet_estados):
        vet_estados, tensoes_neutro = self._aplicar_tensoes_fluxo_direto(
            vet_estados,
            somente_neutros=True,
        )

        self.tensoes_neutro_admitancia = tensoes_neutro

        return vet_estados


    def inicializar_fluxo_direto_completo(self, vet_estados):
        vet_estados, tensoes_completas = self._aplicar_tensoes_fluxo_direto(
            vet_estados,
            somente_neutros=False,
        )

        self.tensoes_fluxo_direto_completo = tensoes_completas

        n_neutros = sum(self.fases[i] == 3 for i in range(len(self.fases)))
        n_fases = len(self.fases) - n_neutros

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

    def identifica_nos_isolados(self):

        def lista_api(valor):
            if callable(valor):
                valor = valor()

            if valor is None:
                return []

            if isinstance(valor, str):
                return [valor] if valor.strip() else []

            return [str(x) for x in valor if str(x).strip()]

        # Ramos e cargas desconectados da fonte
        topologia = self.DSSCircuit.Topology

        ramos = lista_api(topologia.AllIsolatedBranches)
        cargas = lista_api(topologia.AllIsolatedLoads)

        nos_isolados = set()

        # Obtém as barras de cada elemento isolado
        for elemento in ramos + cargas:

            self.DSSCircuit.SetActiveElement(elemento)

            barras = lista_api(
                self.DSSCircuit.ActiveCktElement.BusNames
            )

            for barra_completa in barras:

                partes = barra_completa.lower().split(".")
                nome_barra = partes[0]

                for no in partes[1:]:
                    try:
                        nos_isolados.add((nome_barra, int(no)))
                    except ValueError:
                        pass

        return nos_isolados

    def Calcula_pesos(self, fator_sigma_neutro=1.0) -> tuple:
        inj_pot_at = []
        inj_pot_rat = []
        tensoes = []
        mascara_neutro = []
        for fases, pot_at, pot_rat, tensao in zip(self.barras['Fases'], self.barras['Inj_pot_at'], self.barras['Inj_pot_rat'], self.barras['Tensao']):
            for fase in fases:
                inj_pot_at.append(pot_at[fase])
                inj_pot_rat.append(pot_rat[fase])
                tensoes.append(tensao[fase])
                mascara_neutro.append(fase == 3)

        inj_pot_at = np.array(inj_pot_at)
        inj_pot_rat = np.array(inj_pot_rat)

        idx_medido = np.where(inj_pot_at != 0)[0]

        zP = inj_pot_at[idx_medido]
        zQ = inj_pot_rat[idx_medido]
        zV = np.array(tensoes)

        classe = 0.01
        fs_S = np.percentile(np.abs(np.concatenate([zP, zQ])), 90)

        sP = classe * (np.abs(zP) + fs_S) / 3
        sQ = classe * (np.abs(zQ) + fs_S) / 3
        sV = np.full(len(zV), classe / 3)

        mascara_neutro = np.array(mascara_neutro)

        # Aumentar o peso das tensões dos neutros
        sV[mascara_neutro] *= fator_sigma_neutro

        dp = np.concatenate([sP, sQ, sV])
        pesos = dp**-2
        matriz_pesos = sp.diags(pesos).tocsr()
        
        return matriz_pesos, np.abs(dp)
    
    def passo_kkt(self, H, r, w, n_med, lam=0, eps=0):
        H = H.tocsr()

        Hm = H[:n_med, :]
        Hc = H[n_med:, :]

        rm = np.asarray(r[:n_med]).ravel()
        rc = np.asarray(r[n_med:]).ravel()
        w = np.asarray(w).ravel()

        nx = H.shape[1]
        nc = Hc.shape[0]

        # Matriz normal original
        Gm = (Hm.T @ sp.diags(w) @ Hm).tocsc()

        d = np.asarray(Gm.diagonal())

        dpos = d[np.isfinite(d) & (d > 0)]

        if dpos.size == 0:
            raise RuntimeError("Diagonal de Gm inválida")

        # Escalonamento das variáveis de estado
        piso = 1e-6 * np.median(dpos)

        d_scale = np.maximum(d, piso)

        Dx = sp.diags(
            1.0 / np.sqrt(d_scale),
            format='csc'
        )

        # Matriz normal escalonada
        Gm_s = (Dx @ Gm @ Dx).tocsc()

        # Mantém a regularização relativa original
        Gm_s += lam * sp.identity(nx, format='csc')

        # Lado direito das medidas
        bm = np.asarray(
            Dx @ (Hm.T @ (w * rm))
        ).ravel()

        # Escalonamento das restrições de corrente
        Hc_s = (Hc @ Dx).tocsr()

        # Montagem do KKT escalonado
        K = sp.bmat([
            [Gm_s, Hc_s.T],
            [Hc_s, -eps * sp.identity(nc)]
        ], format='csc')

        b = np.concatenate([bm, rc])

        sol = splu(K).solve(b)

        # Retornar para as coordenadas originais
        delx = Dx @ sol[:nx]

        return np.asarray(delx).ravel()
    
    def run(self, max_error: float, max_iter: int) -> np.array:
        self.iter = 0
        self.residuo = 1
        delx = 1
        while(np.max(np.abs(delx)) > max_error and max_iter > self.iter):
            inicio = time.time()
            self.DSSCircuit.SetActiveBus(self.slack)
            Vr_ref = self.DSSCircuit.Buses.puVoltages[::2]
            Vi_ref = self.DSSCircuit.Buses.puVoltages[1::2]
    
            Vr = np.concatenate([Vr_ref, self.vet_estados[:len(self.vet_estados)//2]])
            Vi = np.concatenate([Vi_ref, self.vet_estados[len(self.vet_estados)//2:]])

            self.residuo, Ir, Ii = self.residuos.calc_res(self.Gs, self.Bs, Vr, Vi)
            fim_res = time.time()

            idx_virtual = self.residuos.idx_virtual
            idx_medido = self.residuos.idx_medido

            J = self.jac.derivadas(self.Gs, self.Bs, Vr, Vi, Ir, Ii, idx_virtual, idx_medido)
            fim_jac = time.time()

            n_med = len(self.residuo) - 2*len(idx_virtual)

            w = self.matriz_pesos.diagonal()
            delx = self.passo_kkt(J, self.residuo, w, n_med)

            self.vet_estados += delx
            fim = time.time()
            
            self.iter += 1
            if np.max(np.abs(self.residuo)) > 1e10:
                raise RuntimeError('Estimador divergiu')

            if np.isnan(self.vet_estados.any()):
                print('NaN na solução')
                break
            if self.iter == 99:
                print('Não convergiu')
            if self.verbose:
                print(f'Os resíduos da iteração {self.iter} levaram {fim_res-inicio:.3f}s')
                print(f'A jacobiana da iteração {self.iter} levou {fim_jac-fim_res:.3f}s')
                print(f'Atualizar o vetor de estados da iteração {self.iter} levou {fim-fim_jac:.3f}')
                print(f'A iteração {self.iter} levou {fim-inicio:.3f}s')

        return self.vet_estados