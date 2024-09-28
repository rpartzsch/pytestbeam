
from numba import njit
import tables as tb
import numpy as np
from numba.typed import List
from numba_progress import ProgressBar

def calculate_device_hit(beam, devices, hit_data, names, folder, log):

    hits_descr = np.dtype([
        ('event_number', '<i8'),
        ('frame', '<u2'),
        ('column', '<u2'),
        ('row', '<u2'),
        ('charge', '<f4')])

    numb_events = beam['nmb_particles']
    device_nmb = len(devices)
    z_positions = List()
    [z_positions.append(i) for i in [dut['z_position'] for dut in devices]]
    device_columns = List()
    [device_columns.append(i) for i in [dut['column'] for dut in devices]]
    device_row = List()
    [device_row.append(i) for i in [dut['row'] for dut in devices]]
    device_columns_pitch = List()
    [device_columns_pitch.append(i) for i in [dut['column_pitch'] for dut in devices]]
    device_row_pitch = List()
    [device_row_pitch.append(i) for i in [dut['row_pitch'] for dut in devices]]
    deltax = List()
    [deltax.append(i) for i in [dut['delta_x'] for dut in devices]]
    deltay = List()
    [deltay.append(i) for i in [dut['delta_y'] for dut in devices]]
    trigger = List()
    [trigger.append(True) if trig == 'triggered' else trigger.append(False) for trig in [dut['trigger'] for dut in devices]]
    accepted_event = np.random.uniform(0, 1, numb_events) < 0.6

    log.info('Calculating particle hit positions')
    for dut in range(device_nmb):
        log.info('Hit positions of %s' %names[dut])
        if trigger[dut] == False:
            hit_table = create_raw_hits(hits_descr, numb_events)
            with ProgressBar(total=numb_events) as progress:
                table = calc_position_untriggered(numb_events, device_nmb, device_row[dut], device_columns[dut], device_columns_pitch[dut],
                            device_row_pitch[dut], deltax[dut], deltay[dut], hit_data, dut, hit_table, 
                            trigger, accepted_event, progress)
        else: 
            hit_table = create_raw_hits(hits_descr, np.sum(accepted_event))
            with ProgressBar(total=numb_events) as progress:
                table = calc_position_triggered(numb_events, device_nmb, device_row[dut], device_columns[dut], device_columns_pitch[dut],
                                device_row_pitch[dut], deltax[dut], deltay[dut], hit_data, dut, hit_table, 
                                trigger, accepted_event, progress)
        table = delete_outs(device_columns[dut], device_row[dut], table)
        create_hit_file(table, folder, names[dut])

@njit
def calc_position(numb_events, device_nmb, row, column, 
                  column_pitch, row_pitch, deltax, deltay, hit_data, dut, hit_table, trigger, accepted_event):

    x = hit_data[3][dut::device_nmb]
    y = hit_data[4][dut::device_nmb]
    event = 0
    for part in range(numb_events):
        if accepted_event[part] == True:
            if trigger[dut] == False:
                hit_table['event_number'][part] = event + 1
                hit_table['column'][part] = ((x[part] + deltax)/column_pitch + column/2) + 1
                hit_table['row'][part] = ((y[part] + deltay)/row_pitch + row/2) + 1
                event += 1
            else:
                hit_table['event_number'][event] = event + 1
                hit_table['column'][event] = ((x[part] + deltax)/column_pitch + column/2) + 1
                hit_table['row'][event] = ((y[part] + deltay)/row_pitch + row/2) + 1
                event += 1
        else:
            if trigger[dut] == False:
                hit_table['event_number'][part] = event + 1
                hit_table['column'][part] = ((x[part] + deltax)/column_pitch + column/2) + 1
                hit_table['row'][part] = ((y[part] + deltay)/row_pitch + row/2) + 1

    return hit_table

# @njit(nogil=True)
def calc_position_untriggered(numb_events, device_nmb, row, column, 
                  column_pitch, row_pitch, deltax, deltay, hit_data, dut, hit_table, trigger, accepted_event, progress_proxy):
    
    x = hit_data[3][dut::device_nmb]
    y = hit_data[4][dut::device_nmb]
    event = 0
    start = 0
    if column_pitch < row_pitch:
        small_pixel = column_pitch
    else:
        small_pixel = row_pitch
    for part in range(numb_events):
        cluster_radius = calc_cluster_radius(hit_data[0][part])
        column_hits, row_hits = calc_cluster_hits(column_pitch, column, deltax, x[part], row_pitch, row, deltay, y[part], cluster_radius, small_pixel)
        cluster_size = len(column_hits)
        stop = start + cluster_size
        hit_table['event_number'][start:stop] = event + 1
        hit_table['column'][start:stop] = column_hits
        hit_table['row'][start:stop] = row_hits
        # hit_table['event_number'][part] = event + 1
        # hit_table['column'][part] = ((x[part] + deltax)/column_pitch + column/2) + 1
        # hit_table['row'][part] = ((y[part] + deltay)/row_pitch + row/2) + 1
        start = stop
        if accepted_event[part] == True:
            event += 1
        progress_proxy.update(1)
    return hit_table

# @njit(nogil=True)
def calc_position_triggered(numb_events, device_nmb, row, column, 
                  column_pitch, row_pitch, deltax, deltay, hit_data, dut, hit_table, trigger, accepted_event, progress_proxy):
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
            cluster_radius = calc_cluster_radius(hit_data[0][part])
            column_hits, row_hits = calc_cluster_hits(column_pitch, column, deltax, x[part], row_pitch, row, deltay, y[part], cluster_radius, small_pixel)
            cluster_size = len(column_hits)
            stop = start + cluster_size
            hit_table['event_number'][start:stop] = event + 1
            hit_table['column'][start:stop] = column_hits
            hit_table['row'][start:stop] = row_hits
            # hit_table['event_number'][event] = event + 1
            # hit_table['column'][event] = ((x[part] + deltax)/column_pitch + column/2) + 1
            # hit_table['row'][event] = ((y[part] + deltay)/row_pitch + row/2) + 1
            event += 1
            start = stop
        progress_proxy.update(1)
    return hit_table

def create_raw_hits(raw_hits_descr, n_events):
    return np.zeros(100*n_events, dtype=raw_hits_descr)

# @njit(nogil=True)
def calc_cluster_radius(energie):
    k = 8.6173324e-5
    T = 300
    distance = 300
    bias = 10
    sigma = distance * np.sqrt(2*k*T/bias) 
    sigma = 10
    return np.abs(np.random.normal(0, sigma))

# @njit(nogil=True)
def calc_cluster_hits(column_pitch, column, deltax, particle_loc_x, row_pitch, row, deltay, particle_loc_y, cluster_radius, small_pixel):

    hits_column = []
    hits_row = []
    last_col = 0
    last_row = 0
    seed_pixel_x = _row_col_from_hit(particle_loc_x, deltax, column_pitch, column)
    seed_pixel_y = _row_col_from_hit(particle_loc_y, deltay, row_pitch, row)
    hits_column.append(int(seed_pixel_x))
    hits_row.append(int(seed_pixel_y))

    for x in range(int(cluster_radius/column_pitch) + 2):
        distance_neg = np.abs(particle_loc_x - _hit_from_row_col(seed_pixel_x + x + 1, deltax, column_pitch, column))
        distance_pos = np.abs(particle_loc_x - _hit_from_row_col(seed_pixel_x - x, deltax, column_pitch, column))

        if distance_pos < cluster_radius:
            hits_column.append(int((particle_loc_x + deltax)/column_pitch + column/2) + 1 + x + 1)
            hits_row.append(int((particle_loc_y + deltay)/row_pitch + row/2) + 1)

        if x > 0:
            if distance_neg < cluster_radius:
                hits_column.append(int((particle_loc_x + deltax)/column_pitch + column/2) + 1 - x - 1)
                hits_row.append(int((particle_loc_y + deltay)/row_pitch + row/2) + 1)

    for y in range(int(cluster_radius/row_pitch) + 2):
        distance_neg = np.abs(particle_loc_y - _hit_from_row_col(seed_pixel_y + y + 1, deltay, row_pitch, row))
        distance_pos = np.abs(particle_loc_y - _hit_from_row_col(seed_pixel_y - y, deltay, row_pitch, row))

        if distance_pos < cluster_radius:
            hits_row.append(int((particle_loc_y + deltay)/row_pitch + row/2) + 1 + y + 1)
            hits_column.append(int((particle_loc_x + deltax)/column_pitch + column/2) + 1)

        if x > 0:
            if distance_neg < cluster_radius:
                hits_row.append(int((particle_loc_y + deltay)/row_pitch + row/2) + 1 - y - 1)
                hits_column.append(int((particle_loc_x + deltax)/column_pitch + column/2) + 1)

    for xy in range(int(cluster_radius/np.sqrt((row_pitch**2 + column_pitch**2))) + 2):
        pass
    for yx in range(int(cluster_radius/np.sqrt((row_pitch**2 + column_pitch**2))) + 2):
        pass
    # for y in range(int(cluster_radius)):
    #     for x in range(int(cluster_radius)):
    #         distance = x**2 + y**2
    #         if distance < cluster_radius**2:
    #             calc_col = int((particle_loc_x + deltax + x)/column_pitch + column/2) + 1
    #             calc_row = int((particle_loc_y + deltay + y)/row_pitch + row/2) + 1
    #             if calc_col != last_col and calc_row != last_row:
    #                 hits_column.append(calc_col)
    #                 hits_row.append(calc_row)
    #                 if x > 0 or y > 0:
    #                     hits_column.append(int((particle_loc_x + deltax - x)/column_pitch + column/2) + 1)
    #                     hits_row.append(int((particle_loc_y + deltay - y)/row_pitch + row/2) + 1)
    #             last_col = calc_col
    #             last_row = calc_row
    return hits_column, hits_row

def delete_outs(column, row, hit_table):
    hit_table = np.delete(hit_table, np.where(hit_table['column']>column))
    hit_table = np.delete(hit_table, np.where(hit_table['row']>row))
    hit_table = np.delete(hit_table, np.where(hit_table['column']<1))
    hit_table = np.delete(hit_table, np.where(hit_table['row']<1))
    return hit_table

def create_hit_file(hit_data, folder, index):
    hits_descr = np.dtype([
        ('event_number', '<i8'),
        ('frame', '<u2'),
        ('column', '<u2'),
        ('row', '<u2'),
        ('charge', '<f4')])

    hit_data = np.array(hit_data, dtype=hits_descr)
    out_file_h5 = tb.open_file(filename=folder + index +'_dut.h5', mode='w')
    output_hits_table = out_file_h5.create_table(where=out_file_h5.root,
                                                    name='Hits',
                                                    description=hits_descr,
                                                    title='Hits for test beam analysis',
                                                    filters=tb.Filters(
                                                    complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))

    output_hits_table.append(hit_data)
    output_hits_table.flush()
    out_file_h5.close()
    return folder + index + '_dut.h5'

def _row_col_from_hit(x, delta, pitch, number):
    return (int((x + delta)/pitch + number/2) + 1)

def _hit_from_row_col(col_row, delta, pitch, number):
    return (pitch*(col_row - number/2 - 1) - delta + pitch/2)