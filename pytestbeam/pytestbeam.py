import matplotlib.pyplot as plt
import numpy as np
import hit
import device
from plotting import correlate, plot_events, plot_default
import yaml
import logger

if __name__ == '__main__':
    log = logger.setup_main_logger('pytestbeam')

    log.info('Preparing simulation')
    with open('setup.yml', "r") as file:
        setup = yaml.full_load(file)

    with open('material.yml', "r") as file:
        material = yaml.full_load(file)

    folder = setup['data_output']
    if folder == None:
        folder = 'output_data/'

    device_material = [setup['deviceses'][dev]['material'] for dev in setup['deviceses']]
    materials = [material[device_material[i]] for i in range(len(device_material))]
    names = [dev for dev in setup['deviceses']]

    devicess = [setup['deviceses'][dev] for dev in setup['deviceses']]
    beam = setup['beam']
    
    hit_tables = hit.tracks(beam, devicess, materials, log)
    if setup['saving_tracks']:
        log.info('Saving tracks this takes forever...')
        hit.create_output_tracks(hit_tables, folder)
    device.calculate_device_hit(beam, devicess, hit_tables, names, folder, log)

    if setup['plotting']:
        plot_default(devicess, names, hit_tables, np.arange(1, 6, 1), folder, log)
