import numpy as np
from scipy.stats import norm

beam = {
    'sigma_x': 2000,
    'sigma_y': 2000,
    'loc_x': 0,
    'loc_y': 0,
    'nmb_particles': 100
}

class sim_initial_hits(object):
    def __init__(self, detector, beam) -> None:
        
        self.column = detector['column']
        self.row = detector['row']
        self.column_pitch = detector['column_pitch']
        self.row_pitch = detector['row_pitch']
        self.thickness = detector['thickness']

        self.bsigmax = beam['sigma_x']
        self.bsigmay = beam['sigma_y']
        self.blocx = beam['loc_x']
        self.blocy = beam['loc_y']
        self.bparticles = beam['nmb_particles']
        
        self.hits_desc = np.dtype([
                            ('event_number', '<i8'),
                            ('frame', '<u2'),
                            ('column', '<u2'),
                            ('row', '<u2'),
                            ('charge', '<f4'),
                            ('timestamp', '<f4')])
        self.hit_table = np.zeros(shape=self.bparticles, dtype=self.hits_desc)

    def hit_distribution(self):

        pdf_x = norm(loc = (self.blocx/self.column_pitch + self.column/2), scale = self.bsigmax/self.column_pitch)
        pdf_y = norm(loc = (self.blocy/self.row_pitch + self.row/2), scale = self.bsigmay/self.row_pitch)

        x_coord = pdf_x.rvs(size = self.bparticles)
        y_coord = pdf_y.rvs(size = self.bparticles)

        return x_coord, y_coord

    def binary_readout(self):
        x_coord, y_coord = self.hit_distribution()

        self.hit_table['column'] = (x_coord)
        self.hit_table['row'] = (y_coord)

        self.hit_table = np.delete(self.hit_table, np.where(self.hit_table['column']>self.column))
        self.hit_table = np.delete(self.hit_table, np.where(self.hit_table['row']>self.row))
        self.hit_table = np.delete(self.hit_table, np.where(self.hit_table['column']<0))
        self.hit_table = np.delete(self.hit_table, np.where(self.hit_table['row']<0))

        return self.hit_table
    
class sim_particle(object):
    def __init__(self, beam) -> None:
        self.energy = [beam['energy']]
        self.angle_x = [float(self.draw_1dgauss(beam['loc_x'], beam['x_disp']))]
        self.angle_y = [float(self.draw_1dgauss(beam['loc_x'], beam['y_disp']))]
        self.x = [float(self.draw_1dgauss(beam['loc_x'], beam['sigma_x']))]
        self.y = [float(self.draw_1dgauss(beam['loc_y'], beam['sigma_y']))]
        self.z = [0]

    def draw_1dgauss(self, loc, sig):
        pdf = norm(loc = loc, scale = sig)
        x = pdf.rvs(size = 1)
        return x

    def draw_2dgauss(self, locx, locy, sigx, sigy):
        pdf_x = norm(loc = locx, scale = sigx)
        pdf_y = norm(loc = locy, scale = sigy)
        x = pdf_x.rvs(size = 1)
        y = pdf_y.rvs(size = 1)
        return x, y

    def highlander_fast_electrons(self, thickness, rad_length, energy):
        beta = 1
        z = -1
        p = energy
        epsilon = thickness/rad_length
        return np.sqrt((13.6/(beta*p)*z)**2*epsilon*(1+0.038*np.log(epsilon))**2)
    
    def scatter(self, scatterer, energy, x_angle, y_angle):
        silic = 93650
        var_angle = self.highlander_fast_electrons(scatterer['thickness'], silic, energy)
        return self.draw_2dgauss(x_angle, y_angle, var_angle, var_angle)
    
    def fly(self, length, x_angle, y_angle):
        deltax = np.tan(x_angle)*length
        deltay = np.tan(y_angle)*length
        return deltax, deltay
    
    def propagate(self, devices):
        for device in devices:
            if self.z[-1] != device['z_position']:
                deltax, deltay = self.fly(device['z_position'], self.angle_x[-1], self.angle_y[-1])
                self.x.append(self.x[-1] + deltax)
                self.y.append(self.y[-1] + deltay)
                self.z.append(device['z_position'])
            angle_x, angle_y = self.scatter(device, self.energy[-1], self.angle_x[-1], self.angle_y[-1])
            self.angle_x.append(float(angle_x))
            self.angle_y.append(float(angle_y))
            self.energy.append(self.energy[-1])
        
    def output_path(self):
        return self.energy, self.angle_x, self.angle_y, self.x, self.y, self.z

