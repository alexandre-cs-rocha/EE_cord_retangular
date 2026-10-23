from pathlib import Path
import numpy as np
import time
import EESD


def get_gabarito(eesd: EESD.EESD) -> np.array:
    Vr = []
    Vi = []
    for barra in eesd.DSSCircuit.Buses:
        Vr = np.concatenate([Vr, barra.puVoltages[::2]])
        Vi = np.concatenate([Vi, barra.puVoltages[1::2]])

    return np.concatenate([Vr[3:], Vi[3:]])

def gabarito_neutro(eesd: EESD.EESD):
    Vr = np.zeros(len(eesd.fases))
    Vi = np.zeros(len(eesd.fases))
    angulos = [0, -2*np.pi/3, 2*np.pi/3]

    i = 0
    for barra, fases in zip(eesd.barras['Nome'], eesd.barras['Fases']):
        for fase in fases:
            if fase == 3:
                eesd.DSSCircuit.SetActiveBus(barra)
                Vr_1 = eesd.DSSCircuit.Buses.puVoltages[::2]
                Vi_1 = eesd.DSSCircuit.Buses.puVoltages[1::2]

                Vr[i] = Vr_1[-1]
                Vi[i] = Vi_1[-1]
                i += 1
                continue
            
            Vr[i] = np.cos(angulos[fase])
            Vi[i] = np.sin(angulos[fase])
            i += 1
    
    vet_estados = np.concatenate((Vr, Vi))
            
    return vet_estados

def erro(gabarito: np.ndarray, vet_estados: np.ndarray):
    erros_abs = np.abs(gabarito - vet_estados)
    idx_max_abs = np.argmax(erros_abs)
    erro_abs = erros_abs[idx_max_abs]

    erros_rel = erros_abs / np.abs(gabarito)
    idx_max_rel = np.argmax(erros_rel)
    erro_rel = erros_rel[idx_max_rel]

    erro_med = erros_abs.mean()

    return erro_abs, idx_max_abs, erro_rel, idx_max_rel, erro_med

def main():
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / 'FRU03' / 'Master_DU06_202412380_FRU03_NTMBSR1PVTTR.dss'
    "'4Bus-YY-Bal' / '4Bus-YY-Bal.dss'"
    "'13Bus' / 'IEEE13Nodeckt.dss'"
    "'123Bus'  / 'IEEE123Master.dss'"
    "'8500-Node' / 'Master.dss'"
    "'ckt7' / 'Master_ckt7.dss'"
    "'FGU04' / 'Master_DU06_202312380_FGU04_NTMBSR1PVTTR.dss'"
    "'FRU03' / 'Master_DU06_202412380_FRU03_NTMBSR1PVTTR.dss'"
    "'FRU07' / 'Master_DU06_202412380_FRU07_NTMBSR1PVTTR.dss'"

    baseva = 1 * 1e5
    verbose = True

    eesd = EESD.EESD(MasterFile, baseva, verbose)

    gabarito = get_gabarito(eesd)
    gabarito_n = gabarito_neutro(eesd)
    #eesd.vet_estados = gabarito.copy()

    inicio = time.time()
    vet_estados = eesd.run(1e-5, 100)
    fim = time.time()

    erro_abs, idx_abs, erro_rel, idx_rel, erro_med = erro(gabarito, vet_estados)

    print(f'Estimador concluido em {fim-inicio}s')
    print(f'O erro médio foi {erro_med:e}')
    print(f'O maior erro absoluto foi {erro_abs:e}')
    print(f'O maior erro relativo foi {erro_rel * 100:.5f}%*')
    print('*Para sistemas com neutro esse valor fica muito alto devido a magnitudes baixas das variáveis')


if __name__ == '__main__':
    main()
