import numpy as np
from numba import njit
from numba.typed import List

def tracks(beam, devices, random_seed=42):

    np.random.seed(random_seed)
    beam_locx = beam['loc_x']
    beam_sigmax = beam['sigma_x']
    beam_locy = beam['loc_y']
    beam_sigmay = beam['sigma_y']
    beam_dispx = beam['x_disp']
    beam_dispy = beam['y_disp']
    energy = List()
    [energy.append(i) for i in [beam['energy']]]
    anglex = List()
    [anglex.append(i) for i in [np.random.normal(0, beam['x_disp'])]]
    angley = List()
    [angley.append(i) for i in [np.random.normal(0, beam['y_disp'])]]
    x = List()
    [x.append(i) for i in [np.random.normal(beam['loc_x'], beam['sigma_x'])]]
    y = List()
    [y.append(i) for i in [np.random.normal(beam['loc_y'], beam['sigma_y'])]]
    z = List()
    [z.append(i) for i in [0]]
    numb_events = beam['nmb_particles']
    device_nmb = len(devices)
    z_positions = List()
    [z_positions.append(i) for i in [dut['z_position'] for dut in devices]]
    scatter_thickness = List()
    [scatter_thickness.append(i) for i in [dut['thickness'] for dut in devices]]


    return generate_tracks(energy, anglex, angley, x, y, z, numb_events, device_nmb, 
                           z_positions, scatter_thickness, 
                           beam_locx, beam_sigmax, beam_locy,
                        beam_sigmay, beam_dispx, beam_dispy, random_seed=42)

@njit
def generate_tracks(energy, anglex, angley, x, y, z, numb_events, device_nmb, z_positions, scatter_thickness, 
                        beam_locx, beam_sigmax, beam_locy,
                        beam_sigmay, beam_dispx, beam_dispy, random_seed=42):
    for events in range(numb_events):
        np.random.seed(events*random_seed)
        for device in range(device_nmb):
            if z[-1] != z_positions[device]:
                deltax, deltay = fly(z_positions[device], anglex[-1], angley[-1])
                x.append(x[-1] + deltax)
                y.append(y[-1] + deltay)
                z.append(z_positions[device])
            angle_x, angle_y = scatter(scatter_thickness[device], energy[-1], anglex[-1], angley[-1])
            anglex.append(angle_x)
            angley.append(angle_y)
            energy.append(energy[-1])
        anglex.append(draw_1dgauss(0, beam_dispx))
        angley.append(draw_1dgauss(0, beam_dispy))
        x.append(draw_1dgauss(beam_locx, beam_sigmax))
        y.append(draw_1dgauss(beam_locy, beam_sigmay))
    return energy, anglex, angley, x, y, z

@njit
def draw_2dgauss(locx, locy, sigx, sigy):
    return np.random.normal(locx, sigx), np.random.normal(locy, sigy)

@njit
def draw_1dgauss(loc, sig):
    return np.random.normal(loc, sig)

@njit
def highlander_fast_electrons(thickness, rad_length, energy):
    beta = 1
    z = -1
    p = energy
    epsilon = thickness/rad_length
    return np.sqrt((13.6/(beta*p)*z)**2*epsilon*(1+0.038*np.log(epsilon))**2)

@njit
def scatter(thickness, energy, x_angle, y_angle):
    silic = 93650
    var_angle = highlander_fast_electrons(thickness, silic, energy)
    return draw_2dgauss(x_angle, y_angle, var_angle, var_angle)

@njit
def fly(length, x_angle, y_angle):
    deltax = np.tan(x_angle)*length
    deltay = np.tan(y_angle)*length
    return deltax, deltay

