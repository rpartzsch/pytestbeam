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
    z_positions = List()
    [z_positions.append(i) for i in [dut["z_position"] for dut in devices]]
    device_columns = List()
    [device_columns.append(i) for i in [dut["column"] for dut in devices]]
    device_row = List()
    [device_row.append(i) for i in [dut["row"] for dut in devices]]
    device_columns_pitch = List()
    [device_columns_pitch.append(i) for i in [dut["column_pitch"] for dut in devices]]
    device_row_pitch = List()
    [device_row_pitch.append(i) for i in [dut["row_pitch"] for dut in devices]]
    deltax = List()
    [deltax.append(i) for i in [dut["delta_x"] for dut in devices]]
    deltay = List()
    [deltay.append(i) for i in [dut["delta_y"] for dut in devices]]
    trigger = List()
    [
        trigger.append(True) if trig == "triggered" else trigger.append(False)
        for trig in [dut["trigger"] for dut in devices]
    ]
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
        cluster_radius = calc_cluster_radius()
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
            small_pixel,
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
            cluster_radius = calc_cluster_radius()
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
                small_pixel,
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
def calc_cluster_radius():
    k = 8.6173324e-5
    T = 300
    distance = 300
    bias = 50
    sigma = distance * np.sqrt(2 * k * T / bias)
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
    small_pixel,
):
    hits_column = []
    hits_row = []
    seed_pixel_x = _row_col_from_hit(particle_loc_x, deltax, column_pitch, column)
    seed_pixel_y = _row_col_from_hit(particle_loc_y, deltay, row_pitch, row)

    if seed_pixel_x > 1 and seed_pixel_x < column:
        if seed_pixel_y > 1 and seed_pixel_y < row:
            for x in range(0, int(cluster_radius / column_pitch) + 2):
                distance_neg = np.abs(
                    particle_loc_x
                    - _hit_from_row_col(
                        seed_pixel_x + x + 1, deltax, column_pitch, column
                    )
                )
                distance_pos = np.abs(
                    particle_loc_x
                    - _hit_from_row_col(seed_pixel_x - x, deltax, column_pitch, column)
                )

                if distance_pos < cluster_radius:
                    col_hit = (
                        int((particle_loc_x + deltax) / column_pitch + column / 2)
                        + 1
                        + x
                    )
                    if col_hit > 1 and col_hit < column:
                        hits_column.append(col_hit)
                        hits_row.append(int(seed_pixel_y))

                if distance_neg < cluster_radius:
                    col_hit = (
                        int((particle_loc_x + deltax) / column_pitch + column / 2) - x
                    )
                    if col_hit > 1 and col_hit < column:
                        hits_column.append(col_hit)
                        hits_row.append(int(seed_pixel_y))

            for y in range(0, int(cluster_radius / row_pitch) + 2):
                distance_neg = np.abs(
                    particle_loc_y
                    - _hit_from_row_col(seed_pixel_y + y + 1, deltay, row_pitch, row)
                )
                distance_pos = np.abs(
                    particle_loc_y
                    - _hit_from_row_col(seed_pixel_y - y, deltay, row_pitch, row)
                )

                if distance_pos < cluster_radius:
                    row_hit = (
                        int((particle_loc_y + deltay) / row_pitch + row / 2) + 1 + y
                    )
                    if row_hit > 1 and row_hit < row:
                        hits_row.append(row_hit)
                        hits_column.append(int(seed_pixel_x))

                if distance_neg < cluster_radius:
                    row_hit = int((particle_loc_y + deltay) / row_pitch + row / 2) - y
                    if row_hit > 1 and row_hit < row:
                        hits_row.append(row_hit)
                        hits_column.append(int(seed_pixel_x))

        for xy in range(
            int(cluster_radius / np.sqrt((row_pitch**2 + column_pitch**2))) + 2
        ):
            distance_neg_y = np.abs(
                particle_loc_y
                - _hit_from_row_col(seed_pixel_y + xy + 1, deltay, row_pitch, row)
            )
            distance_pos_y = np.abs(
                particle_loc_y
                - _hit_from_row_col(seed_pixel_y - xy, deltay, row_pitch, row)
            )
            distance_neg_x = np.abs(
                particle_loc_x
                - _hit_from_row_col(seed_pixel_x + xy + 1, deltax, column_pitch, column)
            )
            distance_pos_x = np.abs(
                particle_loc_x
                - _hit_from_row_col(seed_pixel_x - xy, deltax, column_pitch, column)
            )

            if distance_pos_y**2 + distance_pos_x**2 < cluster_radius**2:
                row_hit = int((particle_loc_y + deltay) / row_pitch + row / 2) + 1 + xy
                col_hit = (
                    int((particle_loc_x + deltax) / column_pitch + column / 2) + 1 + xy
                )
                if row_hit > 1 and row_hit < row:
                    if col_hit > 1 and col_hit < column:
                        hits_row.append(row_hit)
                        hits_column.append(col_hit)

            if distance_neg_y**2 + distance_neg_x**2 < cluster_radius**2:
                row_hit = int((particle_loc_y + deltay) / row_pitch + row / 2) - xy
                col_hit = (
                    int((particle_loc_x + deltax) / column_pitch + column / 2) - xy
                )
                if row_hit > 1 and row_hit < row:
                    if col_hit > 1 and col_hit < column:
                        hits_row.append(row_hit)
                        hits_column.append(col_hit)

        for yx in range(
            int(cluster_radius / np.sqrt((row_pitch**2 + column_pitch**2))) + 2
        ):
            distance_neg_y = np.abs(
                particle_loc_y
                - _hit_from_row_col(seed_pixel_y + yx + 1, deltay, row_pitch, row)
            )
            distance_pos_y = np.abs(
                particle_loc_y
                - _hit_from_row_col(seed_pixel_y - yx, deltay, row_pitch, row)
            )
            distance_neg_x = np.abs(
                particle_loc_x
                - _hit_from_row_col(seed_pixel_x + yx + 1, deltax, column_pitch, column)
            )
            distance_pos_x = np.abs(
                particle_loc_x
                - _hit_from_row_col(seed_pixel_x - yx, deltax, column_pitch, column)
            )

            if distance_pos_y**2 + distance_neg_x**2 < cluster_radius**2:
                row_hit = int((particle_loc_y + deltay) / row_pitch + row / 2) + 1 + yx
                col_hit = (
                    int((particle_loc_x + deltax) / column_pitch + column / 2) - yx
                )
                if row_hit > 1 and row_hit < row:
                    if col_hit > 1 and col_hit < column:
                        hits_row.append(row_hit)
                        hits_column.append(col_hit)

            if distance_neg_y**2 + distance_pos_x**2 < cluster_radius**2:
                row_hit = int((particle_loc_y + deltay) / row_pitch + row / 2) - yx
                col_hit = (
                    int((particle_loc_x + deltax) / column_pitch + column / 2) + yx + 1
                )
                if row_hit > 1 and row_hit < row:
                    if col_hit > 1 and col_hit < column:
                        hits_row.append(row_hit)
                        hits_column.append(col_hit)

    if len(hits_column) > 1:
        res = list(
            set([(hits_column[i], hits_row[i]) for i in range(len(hits_column))])
        )
        col_hits_output = [i[0] for i in res]
        row_hits_output = [i[1] for i in res]
    else:
        col_hits_output = hits_column
        row_hits_output = hits_row
    return col_hits_output, row_hits_output


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
