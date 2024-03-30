from numba import njit
import tables as tb
import numpy as np
from device import device
from hit import sim_initial_hits
from hit import sim_particle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [10,7]
mpl.rcParams['xtick.top'] = True
mpl.rcParams ['xtick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family']= 'DejaVu Serif'
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['figure.facecolor'] = (1,1,1)

from cmcrameri import cm

colormap = cm.navia # color map
darkblue = cm.naviaS.resampled(5)(0) # very dark blue
lightgreen = cm.naviaS.resampled(5)(1) # light green
muddygreen = cm.naviaS.resampled(5)(2) # muddy green
lightblue = cm.naviaS.resampled(5)(3) # light blue
whiteish = cm.naviaS.resampled(6)(4) # whiteish
blue = cm.naviaS.resampled(6)(1) # blue
yellowish = cm.naviaS.resampled(8)(3)


class simulate(object):
    def __init__(self) -> None:
        self.part_desc = np.dtype([
            ('energy', float),
            ('x_angle', float),
            ('y_angle', float),
            ('x', float),
            ('y', float),
            ('z', float)])

    def simulate_hits(self, devices, beam):
        numb_events = beam['nmb_particles']

        numb_devices = len(devices)
        duts = [device(devices[i]) for i in range(numb_devices)]
        hit_tables = [(dut.create_raw_hits(numb_events)) for dut in duts]

        for event in range(numb_events):
            part = sim_particle(beam)
            part.propagate(devices)
            track = part.output_path()
            # hit_dut_1 = np.array(tuple([hit[0] for hit in track]), dtype=self.part_desc)
            # hit_dut_2 = np.array(tuple([hit[1] for hit in track]), dtype=self.part_desc)
            hit_dut = [np.array(tuple([hit[j] for hit in track]), dtype=self.part_desc) for j in range(numb_devices)]

            for i in range(numb_devices):
                hit_tables[i]['event_number'][event] = event + 1
                hit_tables[i]['column'][event] = duts[i].calc_column_position(hit_dut[i]['x']) + 1
                hit_tables[i]['row'][event] = duts[i].calc_row_position(hit_dut[i]['y']) + 1

            # hit_table_2['event_number'][event] = event
            # hit_table_2['column'][event] = duts[1].calc_column_position(hit_dut_2['x'])
            # hit_table_2['row'][event] = duts[1].calc_column_position(hit_dut_2['y'])

        hit_tables = [(duts[i].delete_outs(hit_tables[i])) for i in range(numb_devices)]
        return hit_tables
    

def correlate(file_1, file_2, dut_1, dut_2):
    with tb.open_file(file_1, "r") as in_file:
        dut_hits = in_file.root.Hits[:]

    with tb.open_file(file_2, "r") as in_file:
        ref_hits = in_file.root.Hits[:]


    @njit
    def correlate_position_on_event_number(ref_event_numbers, dut_event_numbers, ref_x_indices, ref_y_indices, dut_x_indices, dut_y_indices, x_corr_hist, y_corr_hist):
        """Correlating the hit/cluster positions on event basis including all permutations.
        The hit/cluster positions are used to fill the X and Y correlation histograms.
        Does the same than the merge of the pandas package:
            df = data_1.merge(data_2, how='left', on='event_number')
            df.dropna(inplace=True)
            correlation_column = np.hist2d(df[column_mean_dut_0], df[column_mean_dut_x])
            correlation_row = np.hist2d(df[row_mean_dut_0], df[row_mean_dut_x])
        The following code is > 10x faster than the above code.
        Parameters
        ----------
        ref_event_numbers: array
            Event number array of the reference DUT.
        dut_event_numbers: array
            Event number array of the second DUT.
        ref_x_indices: array
            X position indices of the refernce DUT.
        ref_y_indices: array
            Y position indices of the refernce DUT.
        dut_x_indices: array
            X position indices of the second DUT.
        dut_y_indices: array
            Y position indices of the second DUT.
        x_corr_hist: array
            X correlation array (2D).
        y_corr_hist: array
            Y correlation array (2D).
        """
        dut_index = 0

        # Loop to determine the needed result array size.astype(np.uint32)
        for ref_index in range(ref_event_numbers.shape[0]):

            while dut_index < dut_event_numbers.shape[0] and dut_event_numbers[dut_index] < ref_event_numbers[ref_index]:  # Catch up with outer loop
                dut_index += 1

            for curr_dut_index in range(dut_index, dut_event_numbers.shape[0]):
                if ref_event_numbers[ref_index] == dut_event_numbers[curr_dut_index]:
                    x_index_ref = ref_x_indices[ref_index]
                    y_index_ref = ref_y_indices[ref_index]
                    x_index_dut = dut_x_indices[curr_dut_index]
                    y_index_dut = dut_y_indices[curr_dut_index]

                    # Add correlation to histogram
                    x_corr_hist[x_index_dut, x_index_ref] += 1
                    y_corr_hist[y_index_dut, y_index_ref] += 1
                else:
                    break

    x_corr_hist, y_corr_hist = np.zeros((dut_1['column'], dut_2['column']), dtype=np.int16), np.zeros((dut_1['row'], dut_2['row']), dtype=np.int32)
    ref_ev = ref_hits["event_number"]
    dut_ev = dut_hits["event_number"]
    ref_x = ref_hits["row"]
    ref_y = ref_hits["column"]
    dut_x = dut_hits["row"]
    dut_y = dut_hits["column"]

    correlate_position_on_event_number(
        ref_event_numbers=ref_ev,
        dut_event_numbers=dut_ev,
        ref_x_indices=ref_x,
        ref_y_indices=ref_y,
        dut_x_indices=dut_x,
        dut_y_indices=dut_y,
        x_corr_hist=x_corr_hist,
        y_corr_hist=y_corr_hist
    )


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 30))

    pcm = ax1.imshow(x_corr_hist, norm=LogNorm(), aspect=0.8, cmap=colormap, origin='lower')
    ax1.grid()
    ax1.set_xlabel('Row DUT 1')
    ax1.set_ylabel('Row DUT 2')
    cbar = fig.colorbar(pcm, ax=ax1, shrink=0.3)
    cbar.set_label('#')
    # plt.savefig("corr_x.pdf")
    # plt.close()

    # ax[1].imshow(y_corr_hist, norm=LogNorm(), aspect=0.66, cmap=colormap, origin='lower')
    pcm = ax2.imshow(y_corr_hist, norm=LogNorm(), aspect=0.8, cmap=colormap, origin='lower')
    ax2.grid()
    ax2.set_xlabel('Column DUT 1')
    ax2.set_ylabel('Column DUT 2')
    
    cbar = fig.colorbar(pcm, ax=ax2, shrink=0.3)
    cbar.set_label('#')


    # plt.imshow(x_corr_hist, norm=LogNorm(), aspect=0.66, cmap=colormap, origin='lower')
    # plt.grid()
    # plt.xlabel('Row DUT 1')
    # plt.ylabel('Row DUT 2')
    # cbar = plt.colorbar()
    # cbar.set_label('#')
    # plt.savefig("corr_x.pdf")
    # plt.close()

    # plt.imshow(y_corr_hist, norm=LogNorm(), aspect=0.66, cmap=colormap, origin='lower')
    # plt.grid()
    # plt.xlabel('Column DUT 1')
    # plt.ylabel('Column DUT 2')
    # # plt.colorbar(aspect=2.05)
    # cbar = plt.colorbar()
    # cbar.set_label('#')
    # plt.savefig("corr_y.pdf")
    # plt.close()