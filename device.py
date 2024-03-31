
from numba import njit
import tables as tb
import numpy as np
from numba.typed import List

def calculate_device_hit(beam, devices, hit_data, folder):

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
    
    for dut in range(device_nmb):
        table = calc_position(numb_events, device_nmb, device_row[dut], device_columns[dut], device_columns_pitch[dut],
                           device_row_pitch[dut], deltax[dut], deltay[dut], hit_data, dut, create_raw_hits(hits_descr, numb_events))
        table = delete_outs(device_columns[dut], device_row[dut], table)
        create_hit_file(table, folder, str(dut))

@njit
def calc_position(numb_events, device_nmb, row, column, 
                  column_pitch, row_pitch, deltax, deltay, hit_data, dut, hit_table):

    x = hit_data[3][dut::device_nmb]
    y = hit_data[4][dut::device_nmb]
    for event in range(numb_events):
        hit_table['event_number'][event] = event + 1
        hit_table['column'][event] = ((x[event] + deltax)/column_pitch + column/2) + 1
        hit_table['row'][event] = ((y[event] + deltay)/row_pitch + row/2) + 1
    return hit_table


def create_raw_hits(raw_hits_descr, n_events):
    return np.zeros(n_events, dtype=raw_hits_descr)

def calc_column_position(deltax, column_pitch, column, hit_pos):
    return ((hit_pos + deltax)/column_pitch + column/2)

def calc_row_position(deltay, row_pitch, row, hit_pos):
    return ((hit_pos + deltay)/row_pitch + row/2)

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