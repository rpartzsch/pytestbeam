from numba import njit
import tables as tb
import numpy as np
from device import device
from hit import sim_initial_hits
from hit import sim_particle


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

        device_1 = device(devices[0])
        device_2 = device(devices[1])

        hit_table_1 = device_1.create_raw_hits(numb_events)
        hit_table_2 = device_2.create_raw_hits(numb_events)

        for event in range(numb_events):
            part = sim_particle(beam)
            part.propagate(devices)
            track = part.output_path()
            hit_dut_1 = np.array(tuple([hit[0] for hit in track]), dtype=self.part_desc)
            hit_dut_2 = np.array(tuple([hit[1] for hit in track]), dtype=self.part_desc)

            hit_table_1['event_number'][event] = event
            hit_table_1['column'][event] = device_1.calc_column_position(hit_dut_1['x'])
            hit_table_1['row'][event] = device_1.calc_row_position(hit_dut_1['y'])

            hit_table_2['event_number'][event] = event
            hit_table_2['column'][event] = device_2.calc_column_position(hit_dut_2['x'])
            hit_table_2['row'][event] = device_2.calc_column_position(hit_dut_2['y'])

        hit_table_1 = device_1.delete_outs(hit_table_1)
        hit_table_2 = device_2.delete_outs(hit_table_2)
        return hit_table_1, hit_table_2
    
