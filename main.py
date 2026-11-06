from pathlib import Path
import numpy as np
import time
import EESD


def get_gabarito(eesd: EESD.EESD, idx_keep: list) -> np.array:
    Vr = []
    Vi = []
    for barra in eesd.DSSCircuit.Buses:
        Vr = np.concatenate([Vr, barra.puVoltages[::2]])
        Vi = np.concatenate([Vi, barra.puVoltages[1::2]])

    return np.concatenate([Vr[idx_keep][3:], Vi[idx_keep][3:]])

def calc_erro(gabarito: np.ndarray, vet_estados: np.ndarray):
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
    MasterFile = CurrentFolder / 'objs' / 'EDPES' / 'Master_DU06_202412380_DMA04_NTMBSR1PVTTR.dss'

    baseva = 1 * 1e5
    verbose = True

    eesd = EESD.EESD(MasterFile, baseva, verbose, inicializar_completo_direto=True)

    gabarito = get_gabarito(eesd, eesd.idx_keep)
    gabarito[gabarito == 0] = 1e-10

    inicio = time.time()
    vet_estados = eesd.run(1e-5, 50)
    fim = time.time()

    erro_abs, _, erro_rel, _, erro_med = calc_erro(gabarito, vet_estados)

    print(f'Estimador concluido em {fim-inicio}s')
    print(f'O erro médio foi {erro_med:e}')
    print(f'O maior erro absoluto foi {erro_abs:e}')
    print(f'O maior erro relativo foi {erro_rel * 100:.5f}%*')
    print('*Para sistemas com neutro esse valor fica muito alto devido a magnitudes baixas das variáveis')


if __name__ == '__main__':
    main()
