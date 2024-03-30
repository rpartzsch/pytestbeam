
from numba import njit
import tables as tb
import numpy as np
from hit import sim_initial_hits
from hit import sim_particle

class device(object):
    def __init__(self, detector) -> None:
        
        self.column = detector['column']
        self.row = detector['row']
        self.column_pitch = detector['column_pitch']
        self.row_pitch = detector['row_pitch']
        self.thickness = detector['thickness']
        self.deltax = detector['delta_x']
        self.deltay = detector['delta_y']

        self.raw_hits_descr = np.dtype([
            ('event_number', '<i8'),
            ('frame', '<u2'),
            ('column', float),
            ('row', float),
            ('charge', '<f4')])

    def create_raw_hits(self, n_events):
        return np.zeros(n_events, dtype=self.raw_hits_descr)

    def calc_column_position(self, hit_pos):
        return ((hit_pos + self.deltax)/self.column_pitch + self.column/2)

    def calc_row_position(self, hit_pos):
        return ((hit_pos + self.deltay)/self.row_pitch + self.row/2)

    def delete_outs(self, hit_table):
        hit_table = np.delete(hit_table, np.where(hit_table['column']>self.column))
        hit_table = np.delete(hit_table, np.where(hit_table['row']>self.row))
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
