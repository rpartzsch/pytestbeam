import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from device import calc_cluster_hits
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit
import logger as logger

import matplotlib as mpl

mpl.rcParams["figure.figsize"] = [10, 7]
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["font.size"] = 12
mpl.rcParams["font.family"] = "DejaVu Serif"
mpl.rcParams["mathtext.default"] = "regular"
mpl.rcParams["errorbar.capsize"] = 3
mpl.rcParams["figure.facecolor"] = (1, 1, 1)

from cmcrameri import cm

colormap = cm.navia  # color map
darkblue = cm.naviaS.resampled(5)(0)  # very dark blue
lightgreen = cm.naviaS.resampled(5)(1)  # light green
muddygreen = cm.naviaS.resampled(5)(2)  # muddy green
lightblue = cm.naviaS.resampled(5)(3)  # light blue
whiteish = cm.naviaS.resampled(6)(4)  # whiteish
blue = cm.naviaS.resampled(6)(1)  # blue
yellowish = cm.naviaS.resampled(8)(3)


def create_injection_array(low, high, bins):
    return np.arange(low, high, (high - low)/bins)

@njit
def create_hits(inj_array, numb_inj, 
                column_pitch, row_pitch, threshold, 
                thickness, 
                column_numb, row_numb, column=0, row=0,):
    inj_hits = np.zeros(np.shape(inj_array)[0])
    for j in range(np.shape(inj_array)[0]):
        energy = inj_array[j] * 1e-6 * 3.6
        hits = 0
        for i in range (numb_inj):
            _, _, charge = calc_cluster_hits(
                column_pitch,
                column_numb,
                column*column_pitch,
                0,
                row_pitch,
                row_numb,
                row*row_pitch,
                0,
                0,
                0,
                threshold,
                thickness,
                energy)
            if charge > [0]:
                hits += 1
        inj_hits[j] = hits / numb_inj
    return inj_hits
            
def create_hit_table(raw_hits_descr: np.dtype, numb_inj: int) -> np.array:
    """Create blank hit table

    Args:
        raw_hits_descr (np.dtype): features of the hit table
        numb_inj (int): number of injections

    Returns:
        np.array: blank hit table
    """
    return np.zeros(numb_inj, dtype=raw_hits_descr)

def plot_s_curve(inj_hits, inj_array, bins):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(inj_array, inj_hits)
    ax.set_xlabel('Injected charge [e⁻]')
    ax.set_ylabel('Hit probability')
    ax.grid()
    return fig

def threshold_scan(dut, low, high, bins, inj_numb, name):
    log.info(f'Threshold scan for {name}')
    inj_array = create_injection_array(low, high, bins)
    hits = create_hits(inj_array, inj_numb, dut['column_pitch'], dut['row_pitch'], 
                        dut['threshold'], dut['thickness'],
                        dut['column'], dut['row'], column=0, row=0)
    s_curve = plot_s_curve(hits, inj_array, bins)
    pdf_pages = PdfPages("../output_data/threshold_scan.pdf")
    pdf_pages.savefig(s_curve)
    pdf_pages.close()
    

if __name__ == '__main__':
    log = logger.setup_main_logger("Threshold scan")
    low = 500
    high = 1500
    bins = 1000
    inj_numb = 100
    with open("../setup.yml", "r") as file:
        setup = yaml.full_load(file)
    name = 'itkpix'
    dut = setup['devices'][name]
    threshold_scan(dut, low, high, bins, inj_numb, name)
