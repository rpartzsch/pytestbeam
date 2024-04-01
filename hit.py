import numpy as np
from numba import njit
from numba.typed import List

def tracks(beam, devices):

    beam_locx = beam['loc_x']
    beam_sigmax = beam['sigma_x']
    beam_locy = beam['loc_y']
    beam_sigmay = beam['sigma_y']
    beam_dispx = beam['x_disp']
    beam_dispy = beam['y_disp']
    beam_anglex = beam['x_angle']
    beam_angley = beam['y_angle']
    particle_distance = 1/beam['particle_rate']*10**9

    energy = List()
    [energy.append(i) for i in [beam['energy']]]
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
    time_stamp = List()
    [time_stamp.append(i) for i in [np.random.poisson(particle_distance)]]
    angle_x, angle_y = scatter(scatter_thickness[0], energy[-1], 0, 0)
    anglex = List()
    [anglex.append(i) for i in [beam['x_angle'] + angle_x]]
    angley = List()
    [angley.append(i) for i in [beam['y_angle'] + angle_y]]

    return generate_tracks(energy, anglex, angley, x, y, z, numb_events, device_nmb, 
                           z_positions, scatter_thickness, 
                           beam_locx, beam_sigmax, beam_locy,
                        beam_sigmay, beam_dispx, beam_dispy, time_stamp, 
                        particle_distance, beam_anglex, beam_angley)

@njit
def generate_tracks(energy, anglex, angley, x, y, z, numb_events, device_nmb, z_positions, scatter_thickness, 
                        beam_locx, beam_sigmax, beam_locy,
                        beam_sigmay, beam_dispx, beam_dispy, time_stamp, particle_distance, beam_anglex, beam_angley, 
                        ):
    for events in range(numb_events):
        for device in range(device_nmb):
            if z[-1] != z_positions[device]:
                deltax, deltay = fly(z_positions[device], anglex[-1], angley[-1])
                x.append(x[-1] + deltax)
                y.append(y[-1] + deltay)
                z.append(z_positions[device])
                angle_x, angle_y = scatter(scatter_thickness[device], energy[-1], 0, 0)
                anglex.append(anglex[-1] + angle_x)
                angley.append(angley[-1] + angle_y)
                energy.append(energy[-1])
                time_stamp.append(time_stamp[-1])
        if events != (numb_events - 1):
            z.append(0)
            energy.append(energy[-1])
            anglex.append(draw_1dgauss(beam_anglex, beam_dispx))
            angley.append(draw_1dgauss(beam_angley, beam_dispy))
            x.append(draw_1dgauss(beam_locx, beam_sigmax))
            y.append(draw_1dgauss(beam_locy, beam_sigmay))
            time_stamp.append(np.random.poisson(particle_distance)+time_stamp[-1])
    return energy, anglex, angley, x, y, z, time_stamp

@njit
def draw_2dgauss(locx, locy, sigx, sigy):
    return np.random.normal(locx, sigx), np.random.normal(locy, sigy)

@njit
def draw_1dgauss(loc, sig):
    return np.random.normal(loc, sig)

@njit
def highlander_fast_electrons(thickness, rad_length, energy):
    z = -1
    p = np.sqrt(energy**2 - 0.511**2)
    beta = np.sqrt(1-1/(1+(energy/0.511)**2))
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

