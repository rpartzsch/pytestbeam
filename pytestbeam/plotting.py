from numba import njit
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [10,7]
mpl.rcParams['xtick.top'] = True
mpl.rcParams ['xtick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['font.size'] = 12
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

from tqdm import tqdm

def plot_default(devices, names, hit_tables, event, folder, log):
    if len(hit_tables[0][0::len(devices)]) > 100000:
        nevents = 100000
    else:
        nevents = len(hit_tables[0][0::len(devices)])
    events = plot_events(devices, names, hit_tables, event, log)
    energy = plot_energy_distribution(names, hit_tables, log, nevents)
    time = plot_times_distribution(names, hit_tables, log, nevents)
    x_angles_last = plot_xangle_distribution(names, hit_tables, log, len(names), nevents)
    y_angles_last = plot_yangle_distribution(names, hit_tables, log, len(names), nevents)
    x_angles_first = plot_xangle_distribution(names, hit_tables, log, 1, nevents)
    y_angles_first = plot_yangle_distribution(names, hit_tables, log, 1, nevents)
    x_last = plot_x_distribution(names, hit_tables, log, len(names), nevents)
    y_last = plot_y_distribution(names, hit_tables, log, len(names), nevents)
    x_first = plot_x_distribution(names, hit_tables, log, 1, nevents)
    y_first = plot_y_distribution(names, hit_tables, log, 1, nevents)

    pdf_pages = PdfPages(folder + 'output_plots.pdf')
    pdf_pages.savefig(events)
    pdf_pages.savefig(energy)
    pdf_pages.savefig(time)
    pdf_pages.savefig(x_first)
    pdf_pages.savefig(y_first)
    pdf_pages.savefig(x_last)
    pdf_pages.savefig(y_last)
    pdf_pages.savefig(x_angles_first)
    pdf_pages.savefig(y_angles_first)
    pdf_pages.savefig(x_angles_last)
    pdf_pages.savefig(y_angles_last)
    pdf_pages.close()

def plot_events(devices, names, hit_tables, event, log):

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    log.info('Plotting example event plot')
    i = 0
    x_line = []
    y_line = []
    z_line = []
    for dut in devices:
        # Define the dimensions of the plane
        x_min, x_max = -dut['column']*dut['column_pitch']/2 + dut['delta_x'], dut['column']*dut['column_pitch']/2 + dut['delta_x']
        y_min, y_max = -dut['row']*dut['row_pitch']/2 + dut['delta_y'], dut['row']*dut['row_pitch']/2 + dut['delta_y']
        # Generate x and y values
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        x, y = np.meshgrid(x, y)

        ax.plot_surface(x, dut['z_position'] , y, color = cm.naviaS.resampled(len(devices))(len(devices)-i), alpha=0.5, label='%s' %names[i])
        i += 1

    x_hits = hit_tables[3]
    y_hits = hit_tables[4]
    for eve in tqdm(event):
        x_line = []
        y_line = []
        z_line = []
        i = 0
        for dut in devices:
            x_line.append(x_hits[i::len(devices)][eve])
            y_line.append(y_hits[i::len(devices)][eve])
            z_line.append(dut['z_position'])
            ax.plot(x_line, z_line, y_line, color='red')
            i += 1

    # Set labels and title
    ax.set_xlabel('x [$\mu$m]')
    ax.set_ylabel('z [$\mu$m]')
    ax.set_zlabel('y [$\mu$m]')
    ax.set_title('Example event plot')
    ax.legend()

    return fig

    # if savefig:
    #     plt.savefig('output_data/example_events.pdf')
    # else:
    #     plt.show()


def plot_energy_distribution(names, hit_tables, log, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info('Plotting energy distribution of first %s events' %events)
    i = 0
    numb_devices = len(names)
    energies = hit_tables[0]
    for dut in tqdm(range(numb_devices)):
        try:
            bin = int(np.std(energies[i::(numb_devices)][:events])*300)
        except:
            bin = 10
        ax.hist(energies[i::numb_devices][:events], bins=bin ,color=cm.naviaS.resampled(numb_devices)(numb_devices-i), label='%s' %names[i], alpha=0.7)
        i += 1

    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('#')
    ax.set_title('Energy distribution after devices')
    ax.legend()
    ax.grid()
    return fig

def plot_times_distribution(names, hit_tables, log, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    i = 0
    numb_devices = len(names)
    log.info('Plotting temporal distribution of first %s events' %events)
    bin_heights, bin_borders, _ = ax.hist(np.subtract(hit_tables[6][0::numb_devices][1:events+1],hit_tables[6][0::numb_devices][:events])*0.001, bins=100 ,color=blue)
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(gauss, bin_centers, bin_heights, p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)])
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(x_interval_for_fit, gauss(x_interval_for_fit, *popt), 
            label='Gauss fit\n A= %.6f\n$\mu$ = %.6f $\mu$s\n$\sigma$ = %.6f $\mu$s'%(popt[0], popt[1], popt[2]), color=lightgreen, linewidth=3)
    ax.set_xlabel('Time [$\mu$s]')
    ax.set_ylabel('#')
    ax.legend()
    ax.set_title('Time distribution')
    ax.grid()
    return fig

def plot_xangle_distribution(names, hit_tables, log, numb_device, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info('Plotting x angle distribution of first %s events on %s' %(events, names[numb_device - 1]))
    i = numb_device - 1
    bin_heights, bin_borders, _ = ax.hist(hit_tables[1][i::len(names)][:events], bins=100 ,color=blue, label='%s' %names[i])
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(gauss, bin_centers, bin_heights, p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)])
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(x_interval_for_fit, gauss(x_interval_for_fit, *popt), 
            label='Gauss fit\n A= %.6f\n$\mu$ = %.6f rad\n$\sigma$ = %.6f rad'%(popt[0], popt[1], popt[2]), color=lightgreen, linewidth=3)
    ax.set_xlabel('x angle [rad]')
    ax.set_ylabel('#')
    ax.set_title('x angle distribution after %s' %names[i])
    ax.legend()
    ax.grid()
    return fig

def plot_yangle_distribution(names, hit_tables, log, numb_device, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info('Plotting y angle distribution of first %s events on %s' %(events, names[numb_device - 1]))
    i = numb_device - 1
    bin_heights, bin_borders, _ = ax.hist(hit_tables[2][i::len(names)][:events], bins=100 ,color=blue, label='%s' %names[i])
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(gauss, bin_centers, bin_heights, p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)])
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(x_interval_for_fit, gauss(x_interval_for_fit, *popt), 
            label='Gauss fit\n A= %.6f\n$\mu$ = %.6f rad\n$\sigma$ = %.6f rad'%(popt[0], popt[1], popt[2]), color=lightgreen, linewidth=3)
    ax.set_xlabel('y angle [rad]')
    ax.set_ylabel('#')
    ax.set_title('y angle distribution after %s' %names[i])
    ax.legend()
    ax.grid()
    return fig

def plot_x_distribution(names, hit_tables, log, numb_device, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info('Plotting x distribution of first %s events on %s' %(events, names[numb_device - 1]))
    i = numb_device - 1
    bin_heights, bin_borders, _ = ax.hist(hit_tables[3][i::len(names)][:events], bins=100 ,color=blue, label='%s' %names[i])
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(gauss, bin_centers, bin_heights, p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)])
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(x_interval_for_fit, gauss(x_interval_for_fit, *popt), 
            label='Gauss fit\n A= %.6f\n$\mu$ = %.6f $\mu$m\n$\sigma$ = %.6f $\mu$m'%(popt[0], popt[1], popt[2]), color=lightgreen, linewidth=3)
    ax.set_xlabel('x [$\mu$m]')
    ax.set_ylabel('#')
    ax.set_title('x distribution after %s' %names[i])
    ax.legend()
    ax.grid()
    return fig

def plot_y_distribution(names, hit_tables, log, numb_device, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info('Plotting y distribution of first %s events on %s' %(events, names[numb_device - 1]))
    i = numb_device - 1
    bin_heights, bin_borders, _ = ax.hist(hit_tables[4][i::len(names)][:events], bins=100 ,color=blue, label='%s' %names[i])
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(gauss, bin_centers, bin_heights, p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)])
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(x_interval_for_fit, gauss(x_interval_for_fit, *popt), 
            label='Gauss fit\n A= %.6f\n$\mu$ = %.6f $\mu$m\n$\sigma$ = %.6f $\mu$m'%(popt[0], popt[1], popt[2]), color=lightgreen, linewidth=3)
    ax.set_xlabel('y [$\mu$m]')
    ax.set_ylabel('#')
    ax.set_title('y distribution after %s' %names[i])
    ax.legend()
    ax.grid()
    return fig


def gauss(x, A, mu, sigma):
    """classic Gaussian function"""
    return (
        A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )



@njit
def centers_from_borders_numba(b):
    centers = np.empty(b.size - 1, np.float64)
    for idx in range(b.size - 1):
        centers[idx] = b[idx] + (b[idx+1] - b[idx]) / 2
    return centers

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