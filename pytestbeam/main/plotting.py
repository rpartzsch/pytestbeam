from numba import njit
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize
import itertools

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

from tqdm import tqdm


def plot_default(devices, names, hit_tables, event, folder, log):
    if len(hit_tables[0][0 :: len(devices)]) > 100000:
        nevents = 100000
    else:
        nevents = len(hit_tables[0][0 :: len(devices)])
    events = plot_events(devices, names, hit_tables, event, log)
    energy = plot_energy_distribution(names, hit_tables, log, nevents)
    device_1 = folder + names[0] + "_dut.h5"
    device_2 = folder + names[-1] + "_dut.h5"
    corr = plot_correlation(
        device_1,
        device_2,
        names[0],
        names[-1],
        log,
        max_cols=(devices[0]['column'] + 1, devices[-1]['column'] + 1),
        max_rows=(devices[0]['row'] + 1, devices[-1]['row'] + 1),
    )
    mean_energy = plot_mean_energy_distribution(
        devices, names, hit_tables, log, nevents
    )
    angle_dispersion = plot_angle_dispersion(
        names, hit_tables, log, len(names), nevents, devices
    )
    time = plot_times_distribution(names, hit_tables, log, nevents)
    x_angles_last = plot_xangle_distribution(
        names, hit_tables, log, len(names), nevents
    )
    y_angles_last = plot_yangle_distribution(
        names, hit_tables, log, len(names), nevents
    )
    x_angles_first = plot_xangle_distribution(names, hit_tables, log, 1, nevents)
    y_angles_first = plot_yangle_distribution(names, hit_tables, log, 1, nevents)
    x_last = plot_x_distribution(names, hit_tables, log, len(names), nevents)
    y_last = plot_y_distribution(names, hit_tables, log, len(names), nevents)
    x_first = plot_x_distribution(names, hit_tables, log, 1, nevents)
    y_first = plot_y_distribution(names, hit_tables, log, 1, nevents)

    pdf_pages = PdfPages(folder + "output_plots.pdf")
    pdf_pages.savefig(events)
    pdf_pages.savefig(energy)
    pdf_pages.savefig(corr)
    pdf_pages.savefig(mean_energy)
    pdf_pages.savefig(angle_dispersion)
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
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    log.info("Plotting example event plot")
    i = 0
    x_line = []
    y_line = []
    z_line = []
    for dut in devices:
        # Define the dimensions of the plane
        x_min, x_max = (
            -dut["column"] * dut["column_pitch"] / 2 + dut["delta_x"],
            dut["column"] * dut["column_pitch"] / 2 + dut["delta_x"],
        )
        y_min, y_max = (
            -dut["row"] * dut["row_pitch"] / 2 + dut["delta_y"],
            dut["row"] * dut["row_pitch"] / 2 + dut["delta_y"],
        )
        # Generate x and y values
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        x, y = np.meshgrid(x, y)

        ax.plot_surface(
            x,
            dut["z_position"],
            y,
            color=cm.naviaS.resampled(len(devices))(len(devices) - i),
            alpha=0.5,
            label="%s" % names[i],
        )
        i += 1

    x_hits = hit_tables[3]
    y_hits = hit_tables[4]
    for eve in tqdm(event):
        x_line = []
        y_line = []
        z_line = []
        i = 0
        for dut in devices:
            x_line.append(x_hits[i :: len(devices)][eve])
            y_line.append(y_hits[i :: len(devices)][eve])
            z_line.append(dut["z_position"])
            ax.plot(x_line, z_line, y_line, color="red")
            i += 1

    # Set labels and title
    ax.set_xlabel("x [$\mu$m]")
    ax.set_ylabel("z [$\mu$m]")
    ax.set_zlabel("y [$\mu$m]")
    ax.set_title("Example event plot")
    ax.legend()

    return fig


def plot_energy_distribution(names, hit_tables, log, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info("Plotting energy distribution of first %s events" % events)
    i = 0
    numb_devices = len(names)
    energies = hit_tables[0]
    for dut in tqdm(range(numb_devices)):
        try:
            bin = int(np.std(energies[i::(numb_devices)][:events]) * 300)
        except:
            bin = 10
        ax.hist(
            energies[i::numb_devices][:events],
            bins=bin,
            color=cm.naviaS.resampled(numb_devices)(numb_devices - i),
            label="%s" % names[i],
            alpha=0.7,
        )
        i += 1

    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("#")
    ax.set_title("Energy distribution after devices")
    ax.legend()
    ax.grid()
    return fig


def plot_mean_energy_distribution(devices, names, hit_tables, log, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info("Plotting mean energy of particles")
    i = 0
    numb_devices = len(names)
    energies = hit_tables[0]
    eners = []
    erners_std = []
    z = []
    for dut in devices:
        eners.append(np.mean(energies[i::(numb_devices)][:events]))
        erners_std.append(np.std(energies[i::(numb_devices)][:events]))
        z.append(dut["z_position"])
        i += 1

    ax.errorbar(z, eners, erners_std, fmt="-o", color=blue)

    ax.set_xlabel(r"Z Position [$\mu m$]")
    ax.set_ylabel("Energy [MeV]")
    ax.set_title("Mean Energy distribution after devices")
    ax.grid()
    return fig


def plot_angle_dispersion(names, hit_tables, log, numb_device, events, devices):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info("Plotting angle dispersion")
    # i = numb_device - 1
    std_x = []
    std_y = []
    z = []
    i = 0
    for dut in devices:
        bin_heights, bin_borders, _ = ax.hist(
            hit_tables[1][i :: len(names)][:events], bins=100
        )
        bin_centers = centers_from_borders_numba(bin_borders)
        popt, _ = optimize.curve_fit(
            gauss,
            bin_centers,
            bin_heights,
            p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)],
        )
        std_x.append(popt[2])
        bin_heights, bin_borders, _ = ax.hist(
            hit_tables[2][i :: len(names)][:events], bins=100
        )
        bin_centers = centers_from_borders_numba(bin_borders)
        popt, _ = optimize.curve_fit(
            gauss,
            bin_centers,
            bin_heights,
            p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)],
        )
        std_y.append(popt[2])
        z.append(dut["z_position"])
        i += 1

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.plot(z, std_x, "-o", color=blue, label="x angle dispersion")
    ax.plot(z, std_y, "-o", color=muddygreen, label="y angle dispersion")
    ax.set_xlabel(r"z position [$\mu m$]")
    ax.set_ylabel(r"$\sigma$ [rad]")
    ax.legend()
    ax.set_title("Angle dispersion after devices")
    ax.grid()
    return fig


def plot_times_distribution(names, hit_tables, log, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    i = 0
    numb_devices = len(names)
    log.info("Plotting temporal distribution of first %s events" % events)
    bin_heights, bin_borders, _ = ax.hist(
        np.subtract(
            hit_tables[6][0::numb_devices][1 : events + 1],
            hit_tables[6][0::numb_devices][:events],
        )
        * 0.001,
        bins=100,
        color=blue,
    )
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(
        gauss,
        bin_centers,
        bin_heights,
        p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)],
    )
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(
        x_interval_for_fit,
        gauss(x_interval_for_fit, *popt),
        label="Gauss fit\n A= %.6f\n$\mu$ = %.6f $\mu$s\n$\sigma$ = %.6f $\mu$s"
        % (popt[0], popt[1], popt[2]),
        color=lightgreen,
        linewidth=3,
    )
    ax.set_xlabel("Time [$\mu$s]")
    ax.set_ylabel("#")
    ax.legend()
    ax.set_title("Time distribution")
    ax.grid()
    return fig


def plot_xangle_distribution(names, hit_tables, log, numb_device, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info(
        "Plotting x angle distribution of first %s events on %s"
        % (events, names[numb_device - 1])
    )
    i = numb_device - 1
    bin_heights, bin_borders, _ = ax.hist(
        hit_tables[1][i :: len(names)][:events],
        bins=100,
        color=blue,
        label="%s" % names[i],
    )
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(
        gauss,
        bin_centers,
        bin_heights,
        p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)],
    )
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(
        x_interval_for_fit,
        gauss(x_interval_for_fit, *popt),
        label="Gauss fit\n A= %.6f\n$\mu$ = %.6f rad\n$\sigma$ = %.6f rad"
        % (popt[0], popt[1], popt[2]),
        color=lightgreen,
        linewidth=3,
    )
    ax.set_xlabel("x angle [rad]")
    ax.set_ylabel("#")
    ax.set_title("x angle distribution after %s" % names[i])
    ax.legend()
    ax.grid()
    return fig


def plot_yangle_distribution(names, hit_tables, log, numb_device, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info(
        "Plotting y angle distribution of first %s events on %s"
        % (events, names[numb_device - 1])
    )
    i = numb_device - 1
    bin_heights, bin_borders, _ = ax.hist(
        hit_tables[2][i :: len(names)][:events],
        bins=100,
        color=blue,
        label="%s" % names[i],
    )
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(
        gauss,
        bin_centers,
        bin_heights,
        p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)],
    )
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(
        x_interval_for_fit,
        gauss(x_interval_for_fit, *popt),
        label="Gauss fit\n A= %.6f\n$\mu$ = %.6f rad\n$\sigma$ = %.6f rad"
        % (popt[0], popt[1], popt[2]),
        color=lightgreen,
        linewidth=3,
    )
    ax.set_xlabel("y angle [rad]")
    ax.set_ylabel("#")
    ax.set_title("y angle distribution after %s" % names[i])
    ax.legend()
    ax.grid()
    return fig


def plot_x_distribution(names, hit_tables, log, numb_device, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info(
        "Plotting x distribution of first %s events on %s"
        % (events, names[numb_device - 1])
    )
    i = numb_device - 1
    bin_heights, bin_borders, _ = ax.hist(
        hit_tables[3][i :: len(names)][:events],
        bins=100,
        color=blue,
        label="%s" % names[i],
    )
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(
        gauss,
        bin_centers,
        bin_heights,
        p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)],
    )
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(
        x_interval_for_fit,
        gauss(x_interval_for_fit, *popt),
        label="Gauss fit\n A= %.6f\n$\mu$ = %.6f $\mu$m\n$\sigma$ = %.6f $\mu$m"
        % (popt[0], popt[1], popt[2]),
        color=lightgreen,
        linewidth=3,
    )
    ax.set_xlabel("x [$\mu$m]")
    ax.set_ylabel("#")
    ax.set_title("x distribution after %s" % names[i])
    ax.legend()
    ax.grid()
    return fig


def plot_y_distribution(names, hit_tables, log, numb_device, events):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    log.info(
        "Plotting y distribution of first %s events on %s"
        % (events, names[numb_device - 1])
    )
    i = numb_device - 1
    bin_heights, bin_borders, _ = ax.hist(
        hit_tables[4][i :: len(names)][:events],
        bins=100,
        color=blue,
        label="%s" % names[i],
    )
    bin_centers = centers_from_borders_numba(bin_borders)
    popt, _ = optimize.curve_fit(
        gauss,
        bin_centers,
        bin_heights,
        p0=[np.max(bin_heights), np.mean(bin_centers), np.std(bin_centers)],
    )
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    ax.plot(
        x_interval_for_fit,
        gauss(x_interval_for_fit, *popt),
        label="Gauss fit\n A= %.6f\n$\mu$ = %.6f $\mu$m\n$\sigma$ = %.6f $\mu$m"
        % (popt[0], popt[1], popt[2]),
        color=lightgreen,
        linewidth=3,
    )
    ax.set_xlabel("y [$\mu$m]")
    ax.set_ylabel("#")
    ax.set_title("y distribution after %s" % names[i])
    ax.legend()
    ax.grid()
    return fig


def gauss(x, A, mu, sigma):
    """classic Gaussian function"""
    return (
        A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


def plot_correlation(
    path_in_device_x,
    path_in_device_y,
    names_1,
    names_2,
    log,
    max_cols=(512, 512),
    max_rows=(512, 512),
    offset=0,
    event_size=50000,
    event_numb_shift=0,
):
    log.info("Plotting correlation")

    if event_size == None:
        with tb.open_file(path_in_device_x, "r") as file:
            table = file.root.Hits[:]
            device_x = np.copy(table)

        with tb.open_file(path_in_device_y, "r") as file:
            table = file.root.Hits[:]
            device_y = np.copy(table)

    else:
        with tb.open_file(path_in_device_x, "r") as file:
            table = file.root.Hits[:]
            device_x = np.copy(table)
            device_x = device_x[
                np.where(
                    (device_x["event_number"] <= offset + event_size)
                    & (offset <= device_x["event_number"])
                )
            ]

        with tb.open_file(path_in_device_y, "r") as file:
            table = file.root.Hits[:]
            device_y = np.copy(table)
            device_y = device_y[
                np.where(
                    (device_y["event_number"] <= offset + event_size)
                    & (offset <= device_y["event_number"])
                )
            ]

    device_y["event_number"] = device_y["event_number"] + event_numb_shift

    device_x = np.delete(
        device_x,
        np.where(np.isin(device_x["event_number"], device_y["event_number"]) == False),
    )
    device_y = np.delete(
        device_y,
        np.where(np.isin(device_y["event_number"], device_x["event_number"]) == False),
    )

    x_corr_hist, y_corr_hist = np.zeros(max_cols, dtype=np.int32), np.zeros(
        max_rows, dtype=np.int32
    )

    buffer_x, buffer_y = _eventloop(device_x, device_y, x_corr_hist, y_corr_hist)

    fig, ax = plt.subplots(1, 2, figsize=(12, 12), constrained_layout=True)
    # fig.tight_layout()
    # axacol = figcol.add_subplot(111)

    # colormap.set_bad((0,0,0))
    im_col = ax[0].imshow(
        buffer_x,
        interpolation="nearest",
        origin="lower",
        cmap=colormap,
        aspect=0.66,
        norm=LogNorm(),
    )

    ax[0].grid()
    ax[0].set_xlabel("Column %s" % names_2)
    ax[0].set_ylabel("Column %s" % names_1)
    # ax[0].colorbar(im_col, ax=ax[0])
    cbar = plt.colorbar(im_col, ax=ax[0], shrink=0.5)
    cbar.set_label("#")

    im_row = ax[1].imshow(
        buffer_y,
        interpolation="nearest",
        origin="lower",
        cmap=colormap,
        aspect=0.66,
        norm=LogNorm(),
    )

    ax[1].grid()
    ax[1].set_xlabel("Row %s" % names_2)
    ax[1].set_ylabel("Row %s" % names_1)
    cbar = plt.colorbar(im_row, ax=ax[1], shrink=0.5)
    cbar.set_label("#")
    fig.suptitle("Correlation first-last device, 50k events", fontsize=16)
    return fig


@njit
def centers_from_borders_numba(b):
    centers = np.empty(b.size - 1, np.float64)
    for idx in range(b.size - 1):
        centers[idx] = b[idx] + (b[idx + 1] - b[idx]) / 2
    return centers


def _eventloop(device_1, device_2, x_hist, y_hist):
    dev_1_ev = device_1["event_number"]
    dev_2_ev = device_2["event_number"]

    dev_1_row = device_1["row"]
    dev_2_row = device_2["row"]

    dev_1_column = device_1["column"]
    dev_2_column = device_2["column"]

    for i in tqdm(range(np.min(dev_1_ev), np.max(dev_1_ev))):
        list1_column = dev_1_column[dev_1_ev == i]
        list2_column = dev_2_column[dev_2_ev == i]
        if len(list1_column) != 0 and len(list2_column) != 0:
            list1_row = dev_1_row[dev_1_ev == i]
            list2_row = dev_2_row[dev_2_ev == i]

            comb_col = list(itertools.product(list1_column, list2_column))
            comb_row = list(itertools.product(list1_row, list2_row))

            for m in range(len(comb_col)):
                x_hist[comb_col[m][0], comb_col[m][1]] += 1

            for m in range(len(comb_row)):
                y_hist[comb_row[m][0], comb_row[m][1]] += 1

    return x_hist, y_hist
