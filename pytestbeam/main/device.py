from numba import njit
import tables as tb
import numpy as np
from numba_progress import ProgressBar


def calculate_device_hit(
    beam: dict, devices: dict, hit_data: list, names: list, folder: str, log
) -> None:
    """Creates HDF5 output hit tables from particle table and device configuration.

    Args:
        beam (dict): Beam parameters (e.g. number of particles, beam profile...)
        devices (dict): Device parameters (e.g. Device pixel pitch, location...)
        hit_data (list): Contains particle hit information
        names (list): Device names for logging and data saving
        folder (str): output folder path
        log (function): loggign function
    """
    hits_descr = np.dtype(
        [
            ("event_number", "<i8"),
            ("frame", "<u2"),
            ("column", "<u2"),
            ("row", "<u2"),
            ("charge", "<f4"),
        ]
    )

    numb_events = beam["nmb_particles"]
    device_nmb = len(devices)
    # Generating device arrays
    trigger = []
    deltay = []
    deltax = []
    device_row_pitch = []
    device_columns_pitch = []
    device_row = []
    device_columns = []
    z_positions = []
    thresholds = []
    thickness = []
    for dut in devices:
        z_positions.append(dut["z_position"])
        device_columns.append(dut["column"])
        device_row.append(dut["row"])
        device_columns_pitch.append(dut["column_pitch"])
        device_row_pitch.append(dut["row_pitch"])
        deltax.append(dut["delta_x"])
        deltay.append(dut["delta_y"])
        trigger.append(dut["trigger"] == "triggered")
        thresholds.append(dut["threshold"])
        thickness.append(dut["thickness"])

    # Trigger and untriggered device hack
    accepted_event = np.random.uniform(0, 1, numb_events) < 0.7

    log.info("Calculating particle hit positions")
    for dut in range(device_nmb):
        log.info("Hit positions of %s" % names[dut])
        # Calculate untriggered device hits
        if trigger[dut]:
            numb_hits = np.sum(accepted_event)
        elif not trigger[dut]:
            numb_hits = numb_events
        hit_table = create_raw_hits(hits_descr, numb_hits)
        with ProgressBar(total=numb_events) as progress:
            table = calc_position(
                numb_events,
                device_nmb,
                device_row[dut],
                device_columns[dut],
                device_columns_pitch[dut],
                device_row_pitch[dut],
                thickness[dut],
                deltax[dut],
                deltay[dut],
                thresholds[dut],
                hit_data,
                dut,
                hit_table,
                accepted_event,
                trigger[dut],
                progress,
            )

        log.info("Saving Data...")
        create_hit_file(table, folder, names[dut])


@njit(nogil=True)
def calc_position(
    numb_events: int,
    device_nmb: int,
    row: int,
    column: int,
    column_pitch: float,
    row_pitch: float,
    thickness: int | float,
    deltax: float,
    deltay: float,
    threshold: float | int,
    hit_data: list,
    dut: int,
    hit_table: np.array,
    accepted_event: list,
    trigger_mode: bool,
    progress_proxy,
) -> np.array:
    """Calculates column and row hits of a device from beam particle hits, using device information.
    The clusters are calculated by approximating the charge cloud from diffusion and Coulomb expansion.

    Args:
        numb_events (int): Total number of particles
        device_nmb (int): Total number of devices
        row (int): Total number of device rows
        column (int): Total number of device column
        column_pitch (float): Column pitch size of the device in um
        row_pitch (float): Row pitch size of the device in um
        deltax (float): Displacement from ideal alignment of the device in x direction in um
        deltay (float): Displacement from ideal alignment of the device in y direction in um
        hit_data (list): Contains particle information
        dut (int): Device number in beam direction
        hit_table (np.array): Output hit table, containing device hit information
        accepted_event (list): Containing information if particle is triggered
        trigger_mode (bool): Information about device trigger mode triggered = True, untriggered = False
        progress_proxy: Counter for proggress bar

    Returns:
        np.array: Output hit table
    """
    x = hit_data[3][dut::device_nmb]
    y = hit_data[4][dut::device_nmb]
    event = 0
    start = 0
    for part in range(numb_events):
        if accepted_event[part] or not trigger_mode:
            energy = hit_data[7][dut][part]
            cluster_radius_x = calc_cluster_radius(energy)
            cluster_radius_y = calc_cluster_radius(energy)
            column_hits, row_hits, charges = calc_cluster_hits(
                column_pitch,
                column,
                deltax,
                x[part],
                row_pitch,
                row,
                deltay,
                y[part],
                cluster_radius_x,
                cluster_radius_y,
                threshold * 1e-3,
                thickness,
                energy,
            )
            cluster_size = len(column_hits)
            if cluster_size > 0:
                stop = start + cluster_size
                hit_table["event_number"][start:stop] = event + 1
                hit_table["column"][start:stop] = column_hits
                hit_table["row"][start:stop] = row_hits
                hit_table["charge"][start:stop] = charges
                start = stop
            if accepted_event[part]:
                event += 1
        progress_proxy.update(1)
    hit_table = hit_table[0:stop]
    return hit_table


def create_raw_hits(raw_hits_descr: np.dtype, n_events: int) -> np.array:
    """Create blank hit table

    Args:
        raw_hits_descr (np.dtype): features of the hit table
        n_events (int): number of particles for size of table

    Returns:
        np.array: blank hit table
    """
    return np.zeros(10 * n_events, dtype=raw_hits_descr)


@njit(nogil=True)
def calc_cluster_radius(energy):
    w_D = 0.8
    w_C = 0.2
    k = 1.38 * 10 ** (-23)  # J/K

    T = 300  # K
    e = 1.602 * 10 ** (-19)  # C
    mu_e = 0.14  # m^2 / Vs
    D = mu_e * k * T / e
    t = 90 * 10 ** (-9)
    diffusion = w_D * np.sqrt(2 * D * t)
    eps_0 = 8.854 * 10 ** (-12)  # As/Vm
    eps_r = 11.7
    E = energy * 1e6  # eV
    Q_tot = e * E / 3.6

    coulomb = w_C * np.cbrt(3 * mu_e * Q_tot * t / (4 * np.pi * eps_0 * eps_r))
    sigma = np.sqrt(diffusion**2 + coulomb**2) * 1e6
    return np.abs(np.random.normal(0, sigma))


@njit(nogil=True)
def calc_cluster_hits(
    column_pitch: float,
    column: int,
    deltax: float,
    particle_loc_x: float,
    row_pitch: float,
    row: int,
    deltay: float,
    particle_loc_y: float,
    cluster_radius_x: float,
    cluster_radius_y: float,
    threshold: int | float,
    thickness: int | float,
    energy: float,
) -> tuple[list, list, list]:
    """Calculates cluster from particle hits and device parameters.

    Args:
        column_pitch (float): Column pitch size of the device in um
        column (int): Total number of device column
        deltax (float): Displacement from ideal alignment of the device in x direction in um
        particle_loc_x (float): x position of the particle hit on the device in um
        row_pitch (float): Row pitch size of the device in um
        row (int): otal number of device rows
        deltay (float): Displacement from ideal alignment of the device in y direction in um
        particle_loc_y (float): y position of the particle hit on the device in um
        cluster_radius (float): Radius of the charge cloud in um

    Returns:
        tuple[list, list, list]: pixel hits of the cluster in the form: [columns, rows, charges]
    """
    hits_column = []
    hits_row = []
    charges = []
    seed_pixel_x = _row_col_from_hit(particle_loc_x, deltax, column_pitch, column)
    seed_pixel_y = _row_col_from_hit(particle_loc_y, deltay, row_pitch, row)
    seed_charge = calc_charge(
        0,
        0,
        energy,
        cluster_radius_x,
        cluster_radius_y,
        row_pitch,
        column_pitch,
        thickness,
    )
    if seed_pixel_x >= 1 and seed_pixel_x <= column:
        if seed_pixel_y >= 1 and seed_pixel_y <= row:
            col_max = _row_col_from_hit(
                particle_loc_x + cluster_radius_x, deltax, column_pitch, column
            )
            col_min = _row_col_from_hit(
                particle_loc_x - cluster_radius_x, deltax, column_pitch, column
            )
            row_max = _row_col_from_hit(
                particle_loc_y + cluster_radius_y, deltay, row_pitch, row
            )
            row_min = _row_col_from_hit(
                particle_loc_y - cluster_radius_y, deltay, row_pitch, row
            )
            for cols in range(col_min, col_max + 1):
                for rows in range(row_min, row_max + 1):
                    x_test = _hit_from_row_col(
                        cols, delta=deltax, pitch=column_pitch, number=column
                    )
                    y_test = _hit_from_row_col(
                        rows, delta=deltay, pitch=row_pitch, number=row
                    )
                    charge = (
                        calc_charge(
                            x_test - particle_loc_x,
                            y_test - particle_loc_y,
                            energy,
                            cluster_radius_x,
                            cluster_radius_y,
                            row_pitch,
                            column_pitch,
                            thickness,
                        )
                        * column_pitch
                        * row_pitch
                    )
                    if charge >= threshold:
                        if cluster_radius_x > 0 and cluster_radius_y > 0:
                            if (x_test - particle_loc_x) ** 2 / cluster_radius_x**2 + (
                                y_test - particle_loc_y
                            ) ** 2 / cluster_radius_y**2 <= 1:
                                if cols >= 1 and cols <= column:
                                    if rows >= 1 and rows <= row:
                                        hits_column.append(cols)
                                        hits_row.append(rows)
                                        charges.append(charge)
            if len(hits_column) == 0 and seed_charge >= threshold:
                hits_column.append(seed_pixel_x)
                hits_row.append(seed_pixel_y)
                charges.append(seed_charge)
    return hits_column, hits_row, charges


def create_hit_file(hit_data: tb.table, folder: str, index: str) -> str:
    """Creates outputs HDF5 file

    Args:
        hit_data (tb.table): data table
        folder (str): output folder path
        index (str): describing index as additional file name

    Returns:
        str: output path and file name
    """
    hits_descr = np.dtype(
        [
            ("event_number", "<i8"),
            ("frame", "<u2"),
            ("column", "<u2"),
            ("row", "<u2"),
            ("charge", "<f4"),
        ]
    )

    hit_data = np.array(hit_data, dtype=hits_descr)
    out_file_h5 = tb.open_file(filename=folder + index + "_dut.h5", mode="w")
    output_hits_table = out_file_h5.create_table(
        where=out_file_h5.root,
        name="Hits",
        description=hits_descr,
        title="Hits for test beam analysis",
        filters=tb.Filters(complib="blosc", complevel=5, fletcher32=False),
    )

    output_hits_table.append(hit_data)
    output_hits_table.flush()
    out_file_h5.close()
    return folder + index + "_dut.h5"


@njit(nogil=True)
def _row_col_from_hit(x: float, delta: float, pitch: float, number: int) -> int:
    """Calculates the row or column hit from the particle position x or y

    Args:
        x (float): particle position
        delta (float): displacement of the device
        pitch (float): pitch of the columns or rows
        number (int): total number of rows or columns

    Returns:
        int: column or row index
    """
    return int((x + delta) / pitch + number / 2) + 1


@njit(nogil=True)
def _hit_from_row_col(col_row: int, delta: float, pitch: float, number: int) -> float:
    """Calculates the middle position of a pixel row or column

    Args:
        col_row (int): column or row index
        delta (float): displacement of the device
        pitch (float): pitch of the columns or rows
        number (int): total number of rows or columns

    Returns:
        float: middle position of the pixel in x or y
    """
    return pitch * (col_row - number / 2 - 1) - delta + pitch / 2


@njit(nogil=True)
def gauss(x, mu, sigma):
    """classic Gaussian function"""
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


@njit(nogil=True)
def calc_charge(
    x,
    y,
    energy,
    cluster_radius_x,
    cluster_radius_y,
    row_pitch,
    column_pitch,
    thickness,
    noise=True,
):
    E = energy * 1e6  # eV
    Q_tot = E / 3.6

    if noise:
        noise = draw_noise_charge(300, row_pitch, column_pitch, thickness)
    else:
        noise = 0
    if cluster_radius_x < 1:
        cluster_radius_x = 1 / np.sqrt(2 * np.pi)
    if cluster_radius_y < 1:
        cluster_radius_y = 1 / np.sqrt(2 * np.pi)
    signal = Q_tot * (gauss(x, 0, cluster_radius_x) * gauss(y, 0, cluster_radius_y))
    return (signal + noise) * 1e-3


@njit
def draw_noise_charge(temperature, row_pitch, column_pitch, thickness):
    k_b = 1.38 * 10 ** (-23)  # J/K
    e = 1.602 * 10 ** (-19)  # C
    eps_0 = 8.854 * 10 ** (-12)  # As/Vm
    eps_r = 11.7
    C = eps_0 * eps_r * row_pitch * column_pitch / thickness * 1e-6
    sigma_kbt = np.sqrt(k_b * temperature * C) / e
    noise = np.random.normal(0, sigma_kbt)
    return noise
