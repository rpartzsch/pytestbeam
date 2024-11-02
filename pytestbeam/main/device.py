from numba import njit
import tables as tb
import numpy as np
from numba.typed import List
from numba_progress import ProgressBar


def calculate_device_hit(beam, devices, hit_data, names, folder, log):
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
    trigger = List()
    deltay = List()
    deltax = List()
    device_row_pitch = List()
    device_columns_pitch = List()
    device_row = List()
    device_columns = List()
    z_positions = List()
    for dut in devices:
        z_positions.append(dut["z_position"])
        device_columns.append(dut["column"])
        device_row.append(dut["row"])
        device_columns_pitch.append(dut["column_pitch"])
        device_row_pitch.append(dut["row_pitch"])
        deltax.append(dut["delta_x"])
        deltay.append(dut["delta_y"])
        trigger.append(dut["trigger"] == "triggered")

    # Trigger and untriggered device hack
    accepted_event = np.random.uniform(0, 1, numb_events) < 0.7

    log.info("Calculating particle hit positions")
    for dut in range(device_nmb):
        log.info("Hit positions of %s" % names[dut])
        if trigger[dut] == False:
            hit_table = create_raw_hits(hits_descr, numb_events)
            with ProgressBar(total=numb_events) as progress:
                table = calc_position_untriggered(
                    numb_events,
                    device_nmb,
                    device_row[dut],
                    device_columns[dut],
                    device_columns_pitch[dut],
                    device_row_pitch[dut],
                    deltax[dut],
                    deltay[dut],
                    hit_data,
                    dut,
                    hit_table,
                    trigger,
                    accepted_event,
                    progress,
                )
        else:
            hit_table = create_raw_hits(hits_descr, np.sum(accepted_event))
            with ProgressBar(total=numb_events) as progress:
                table = calc_position_triggered(
                    numb_events,
                    device_nmb,
                    device_row[dut],
                    device_columns[dut],
                    device_columns_pitch[dut],
                    device_row_pitch[dut],
                    deltax[dut],
                    deltay[dut],
                    hit_data,
                    dut,
                    hit_table,
                    trigger,
                    accepted_event,
                    progress,
                )
        # table = delete_outs(device_columns[dut], device_row[dut], table)
        log.info("Saving Data...")
        create_hit_file(table, folder, names[dut])


@njit
def calc_position(
    numb_events,
    device_nmb,
    row,
    column,
    column_pitch,
    row_pitch,
    deltax,
    deltay,
    hit_data,
    dut,
    hit_table,
    trigger,
    accepted_event,
):
    x = hit_data[3][dut::device_nmb]
    y = hit_data[4][dut::device_nmb]
    event = 0
    for part in range(numb_events):
        if accepted_event[part] == True:
            if trigger[dut] == False:
                hit_table["event_number"][part] = event + 1
                hit_table["column"][part] = (
                    (x[part] + deltax) / column_pitch + column / 2
                ) + 1
                hit_table["row"][part] = ((y[part] + deltay) / row_pitch + row / 2) + 1
                event += 1
            else:
                hit_table["event_number"][event] = event + 1
                hit_table["column"][event] = (
                    (x[part] + deltax) / column_pitch + column / 2
                ) + 1
                hit_table["row"][event] = ((y[part] + deltay) / row_pitch + row / 2) + 1
                event += 1
        else:
            if trigger[dut] == False:
                hit_table["event_number"][part] = event + 1
                hit_table["column"][part] = (
                    (x[part] + deltax) / column_pitch + column / 2
                ) + 1
                hit_table["row"][part] = ((y[part] + deltay) / row_pitch + row / 2) + 1

    return hit_table


@njit(nogil=True)
def calc_position_untriggered(
    numb_events,
    device_nmb,
    row,
    column,
    column_pitch,
    row_pitch,
    deltax,
    deltay,
    hit_data,
    dut,
    hit_table,
    trigger,
    accepted_event,
    progress_proxy,
):
    x = hit_data[3][dut::device_nmb]
    y = hit_data[4][dut::device_nmb]
    event = 0
    start = 0
    if column_pitch < row_pitch:
        small_pixel = column_pitch
    else:
        small_pixel = row_pitch
    for part in range(numb_events):
        energy = hit_data[7][dut][part]
        cluster_radius = calc_cluster_radius(energy)
        column_hits, row_hits = calc_cluster_hits(
            column_pitch,
            column,
            deltax,
            x[part],
            row_pitch,
            row,
            deltay,
            y[part],
            cluster_radius,
        )
        cluster_size = len(column_hits)
        if cluster_size > 0:
            stop = start + cluster_size
            hit_table["event_number"][start:stop] = event + 1
            hit_table["column"][start:stop] = column_hits
            hit_table["row"][start:stop] = row_hits
            start = stop
        if accepted_event[part] == True:
            event += 1
        progress_proxy.update(1)
    hit_table = hit_table[0:stop]
    return hit_table


@njit(nogil=True)
def calc_position_triggered(
    numb_events,
    device_nmb,
    row,
    column,
    column_pitch,
    row_pitch,
    deltax,
    deltay,
    hit_data,
    dut,
    hit_table,
    trigger,
    accepted_event,
    progress_proxy,
):
    x = hit_data[3][dut::device_nmb]
    y = hit_data[4][dut::device_nmb]
    event = 0
    start = 0
    if column_pitch < row_pitch:
        small_pixel = column_pitch
    else:
        small_pixel = row_pitch
    for part in range(numb_events):
        if accepted_event[part] == True:
            energy = hit_data[7][dut][part]
            cluster_radius = calc_cluster_radius(energy)
            column_hits, row_hits = calc_cluster_hits(
                column_pitch,
                column,
                deltax,
                x[part],
                row_pitch,
                row,
                deltay,
                y[part],
                cluster_radius,
            )
            cluster_size = len(column_hits)
            if cluster_size > 0:
                stop = start + cluster_size
                hit_table["event_number"][start:stop] = event + 1
                hit_table["column"][start:stop] = column_hits
                hit_table["row"][start:stop] = row_hits
                start = stop
            event += 1
        progress_proxy.update(1)
    hit_table = hit_table[0:stop]
    return hit_table


def create_raw_hits(raw_hits_descr, n_events):
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
    column_pitch,
    column,
    deltax,
    particle_loc_x,
    row_pitch,
    row,
    deltay,
    particle_loc_y,
    cluster_radius,
):
    hits_column = []
    hits_row = []
    seed_pixel_x = _row_col_from_hit(particle_loc_x, deltax, column_pitch, column)
    seed_pixel_y = _row_col_from_hit(particle_loc_y, deltay, row_pitch, row)
    if seed_pixel_x >= 1 and seed_pixel_x <= column:
        if seed_pixel_y >= 1 and seed_pixel_y <= row:
            col_max = _row_col_from_hit(
                particle_loc_x + cluster_radius, deltax, column_pitch, column
            )
            col_min = _row_col_from_hit(
                particle_loc_x - cluster_radius, deltax, column_pitch, column
            )
            row_max = _row_col_from_hit(
                particle_loc_y + cluster_radius, deltay, row_pitch, row
            )
            row_min = _row_col_from_hit(
                particle_loc_y - cluster_radius, deltay, row_pitch, row
            )
            for cols in range(col_min, col_max + 1):
                for rows in range(row_min, row_max + 1):
                    x_test = _hit_from_row_col(
                        cols, delta=deltax, pitch=column_pitch, number=column
                    )
                    y_test = _hit_from_row_col(
                        rows, delta=deltay, pitch=row_pitch, number=row
                    )
                    if (x_test - particle_loc_x) ** 2 + (
                        y_test - particle_loc_y
                    ) ** 2 <= cluster_radius**2:
                        if cols >= 1 and cols <= column:
                            if rows >= 1 and rows <= row:
                                hits_column.append(cols)
                                hits_row.append(rows)
            if len(hits_column) == 0:
                hits_column.append(seed_pixel_x)
                hits_row.append(seed_pixel_y)
    return hits_column, hits_row


def delete_outs(column, row, hit_table):
    hit_table = np.delete(hit_table, np.where(hit_table["column"] > column))
    hit_table = np.delete(hit_table, np.where(hit_table["row"] > row))
    hit_table = np.delete(hit_table, np.where(hit_table["column"] < 1))
    hit_table = np.delete(hit_table, np.where(hit_table["row"] < 1))
    return hit_table


def create_hit_file(hit_data, folder, index):
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
def _row_col_from_hit(x, delta, pitch, number):
    return int((x + delta) / pitch + number / 2) + 1


@njit(nogil=True)
def _hit_from_row_col(col_row, delta, pitch, number):
    return pitch * (col_row - number / 2 - 1) - delta + pitch / 2
