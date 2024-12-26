import numpy as np
import yaml

import main.hit as hit
import main.device as device
from main.plotting import plot_default
import main.logger as logger

if __name__ == "__main__":
    log = logger.setup_main_logger("pytestbeam")

    log.info("Preparing simulation")
    with open("setup.yml", "r") as file:
        setup = yaml.full_load(file)

    with open("material.yml", "r") as file:
        material = yaml.full_load(file)

    FOLDER = setup["data_output"]
    if FOLDER is None:
        FOLDER = "output_data/"

    device_material = [
        setup["deviceses"][dev]["material"] for dev in setup["deviceses"]
    ]
    materials = [material[device_material[i]] for i in range(len(device_material))]
    names = [dev for dev in setup["deviceses"]]

    devices = [setup["deviceses"][dev] for dev in setup["deviceses"]]
    beam = setup["beam"]

    hit_tables = hit.tracks(beam, devices, materials, log)
    if setup["saving_tracks"]:
        log.info("Saving tracks this takes forever...")
        hit.create_output_tracks(hit_tables, FOLDER)
    device.calculate_device_hit(beam, devices, hit_tables, names, FOLDER, log)

    if setup["plotting"]:
        plot_default(devices, names, hit_tables, np.arange(1, 6, 1), FOLDER, log)
