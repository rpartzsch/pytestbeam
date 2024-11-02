import numpy as np
from numba import njit
from numba.typed import List
from numba_progress import ProgressBar
import tables as tb


def tracks(
    beam: dict, devices: dict, materials: dict, log
) -> tuple[list, list, list, list, list, list, list]:
    """Create test beam setup and generate particle tracks. prepares calculation of particle tracks for Numba.

    Args:
        beam (dict): Consists of the beam parameters like particle energy, dispersion and particle rate
        devices (dict): Consists of the device setup geometry, material budget usw.
        materials (dict): Device materials, this is used to calculate the multiple scattering distribution.
        log (function): logging function

    Returns:
        tuple[list, list, list, list, list, list, list]: A list of parameters for each particle. The lists consist of: energy, anglex, angley, x, y, z, time_stamp, energy_lost
    """
    log.info("Creating telescope setup")
    # reading configuration
    beam_locx = beam["loc_x"]
    beam_sigmax = beam["sigma_x"]
    beam_locy = beam["loc_y"]
    beam_sigmay = beam["sigma_y"]
    beam_dispx = beam["x_disp"]
    beam_dispy = beam["y_disp"]
    beam_anglex = beam["x_angle"]
    beam_angley = beam["y_angle"]
    beam_energy = beam["energy"]
    particle_distance = 1 / beam["particle_rate"] * 10**9
    device_nmb = len(devices)
    numb_events = beam["nmb_particles"]

    # Generating material budgets
    Z = List()
    A = List()
    rho = List()
    rad_length = List()
    for material in materials:
        Z.append(material["Z"])
        A.append(material["A"])
        rho.append(material["rho"])
        rad_length.append(material["rad_length"])

    # Generating duts
    x_extend_pos = List()
    x_extend_neg = List()
    y_extend_pos = List()
    y_extend_neg = List()
    z_positions = List()
    scatter_thickness = List()
    for dut in devices:
        x_extend_pos.append(dut["column_pitch"] * dut["column"] / 2 + dut["delta_x"])
        x_extend_neg.append(-dut["column_pitch"] * dut["column"] / 2 + dut["delta_x"])
        y_extend_pos.append(dut["row_pitch"] * dut["row"] / 2 + dut["delta_y"])
        y_extend_neg.append(-dut["row_pitch"] * dut["row"] / 2 + dut["delta_y"])
        z_positions.append(dut["z_position"])
        scatter_thickness.append(dut["thickness"])

    # Generating starting event
    x = List()
    y = List()
    z = List()
    time_stamp = List()
    anglex = List()
    angley = List()
    x.append(np.random.normal(beam["loc_x"], beam["sigma_x"]))
    y.append(np.random.normal(beam["loc_y"], beam["sigma_y"]))
    z.append(0)
    time_stamp.append(np.random.poisson(particle_distance))

    angle_x, angle_y = scatter(
        scatter_thickness[0], beam["energy"], 0, 0, rad_length[0]
    )

    anglex.append(beam["x_angle"] + angle_x)
    angley.append(beam["y_angle"] + angle_y)

    log.info("Generating particle tracks")
    # Generating energy loss in each device
    energy_lost = np.zeros((device_nmb, numb_events))
    for dev in range(device_nmb):
        args = (
            scatter_thickness[dev] * 10 ** (-4),
            -1,
            Z[dev],
            A[dev],
            (beam["energy"] - np.mean(energy_lost[dev - 1])),
            rho[dev],
            1,
        )
        energy_lost[dev] = sample_dist_fast(
            landau_approx,
            args,
            n=numb_events,
            xmin=0,
            xmax=0.5,
            mode="ntrue",
        )
    energy = List()
    energy.append(beam["energy"] - energy_lost[0][0])

    with ProgressBar(total=numb_events) as progress:
        energy, anglex, angley, x, y, z, time_stamp = generate_tracks(
            energy,
            anglex,
            angley,
            x,
            y,
            z,
            numb_events,
            device_nmb,
            z_positions,
            scatter_thickness,
            beam_locx,
            beam_sigmax,
            beam_locy,
            beam_sigmay,
            beam_dispx,
            beam_dispy,
            time_stamp,
            particle_distance,
            beam_anglex,
            beam_angley,
            energy_lost,
            beam_energy,
            x_extend_pos,
            x_extend_neg,
            y_extend_pos,
            y_extend_neg,
            rad_length,
            progress,
        )

    return energy, anglex, angley, x, y, z, time_stamp, energy_lost


@njit(nogil=True)
def generate_tracks(
    energy: list,
    anglex: list,
    angley: list,
    x: list,
    y: list,
    z: list,
    numb_events: list,
    device_nmb: list,
    z_positions: list,
    scatter_thickness: list,
    beam_locx: list,
    beam_sigmax: list,
    beam_locy: list,
    beam_sigmay: list,
    beam_dispx: list,
    beam_dispy: list,
    time_stamp: list,
    particle_distance: list,
    beam_anglex: list,
    beam_angley: list,
    energy_lost: list,
    beam_energy: list,
    x_extend_pos: list,
    x_extend_neg: list,
    y_extend_pos: list,
    y_extend_neg: list,
    rad_length: list,
    progress_proxy,
) -> tuple[list, list, list, list, list, list, list]:
    """Generate each individual particle track. The energy lost is calculated from a Landau distribution.
        Each particles scatters according to a Gaussian distribution, where the width originates from the Higlander formula.
        The function is called with initial starting positions for event zero and creates the full parameter lists.

    Args:
        energy (list): List containing containing the energy of each particle
        anglex (list): x angles of each particle
        angley (list): y angles of each particle
        x (list): x position of each particle at each scattering device
        y (list): y position of each particle at each scattering device
        z (list): z position of each particle at each scattering device
        numb_events (list): Total number of events
        device_nmb (list): Total number of scattering devices
        z_positions (list): Z positions of each device
        scatter_thickness (list): Scattering thickness of each device
        beam_locx (list): Beam spot position in x
        beam_sigmax (list): Beam spot width in x
        beam_locy (list): Beam spot position in y
        beam_sigmay (list): Beam spot width in x
        beam_dispx (list): Beam dispersion in x
        beam_dispy (list): Beam dispersion in y
        time_stamp (list): Time stamp of each particle
        particle_distance (list): Temporal distance between individual particles. This is used to calculate the time stamp of each particle
        beam_anglex (list): Beam angle in x
        beam_angley (list): Beam angle in y
        energy_lost (list): Energy lost of each particle at each device
        beam_energy (list): Initial beam energy
        x_extend_pos (list): Boarders of each device in x
        x_extend_neg (list): Boarders of each device in -x
        y_extend_pos (list): Boarders of each device in y
        y_extend_neg (list): Boarders of each device in -y
        rad_length (list): radiation length of each scattering device
        progress_proxy (function): ProgressBar for Numba

    Returns:
        tuple[list, list, list, list, list, list, list]: A list of parameters for each particle. The lists consist of: energy, anglex, angley, x, y, z, time_stamp
    """
    for events in range(numb_events):
        for device in range(device_nmb):
            if z[-1] != z_positions[device]:
                deltax, deltay = fly(z_positions[device], anglex[-1], angley[-1])
                x.append(x[-1] + deltax)
                y.append(y[-1] + deltay)
                z.append(z_positions[device])
                angle_x, angle_y = scatter(
                    scatter_thickness[device], energy[-1], 0, 0, rad_length[device]
                )
                if (
                    x[-1] > x_extend_pos[device]
                    or x[-1] < x_extend_neg[device]
                    or y[-1] > y_extend_pos[device]
                    or y[-1] < y_extend_neg[device]
                ):
                    angle_x = 0
                    angle_y = 0
                    energy_lost[device][events] = 0
                anglex.append(anglex[-1] + angle_x)
                angley.append(angley[-1] + angle_y)
                energy.append(energy[-1] - energy_lost[device][events])
                time_stamp.append(time_stamp[-1])
        if events != (numb_events - 1):
            z.append(0)
            energy.append(beam_energy - energy_lost[0][events])
            anglex.append(draw_1dgauss(beam_anglex, beam_dispx))
            angley.append(draw_1dgauss(beam_angley, beam_dispy))
            x.append(draw_1dgauss(beam_locx, beam_sigmax))
            y.append(draw_1dgauss(beam_locy, beam_sigmay))
            time_stamp.append(np.random.poisson(particle_distance) + time_stamp[-1])
        progress_proxy.update(1)
    return energy, anglex, angley, x, y, z, time_stamp


@njit
def draw_2dgauss(
    locx: float, locy: float, sigx: float, sigy: float
) -> tuple[float, float]:
    """Two dimensional Gaussian draw

    Args:
        locx (float): mean position of first Gauss distribution
        locy (float): width of first Gauss distribution
        sigx (float): mean position of second Gauss distribution
        sigy (float): width of second Gauss distribution

    Returns:
        tuple[float, float]: Draw of the two Gaussian distributions
    """
    return np.random.normal(locx, sigx), np.random.normal(locy, sigy)


@njit
def draw_1dgauss(loc: float, sig: float) -> float:
    """One dimensional Gaussian dra

    Args:
        loc (float): mean position of Gauss distribution
        sig (float): width of Gauss distribution

    Returns:
        float: Draw from Gaussian distribution
    """
    return np.random.normal(loc, sig)


@njit
def highlander_fast_electrons(
    thickness: float, rad_length: float, energy: float
) -> float:
    """Generate width of Gaussian distribution for multiple scattering from Highander formula.

    Args:
        thickness (float): thickness of the multiple scatterer in um
        rad_length (float): Radiation length of each scatterer in um
        energy (float): Energy of the traversing particle in MeV

    Returns:
        float: Mean scattering angle
    """
    z = -1
    # p = np.sqrt(energy**2 - 0.511**2)
    p = energy
    beta = np.sqrt(1 - 1 / (1 + (energy / 0.511) ** 2))
    epsilon = thickness / rad_length
    return np.sqrt(
        (13.6 / (beta * p) * z) ** 2 * epsilon * (1 + 0.038 * np.log(epsilon)) ** 2
    )


@njit
def scatter(
    thickness: float, energy: float, x_angle: float, y_angle: float, rad_length: float
) -> tuple[float, float]:
    """Calculate scatterign angle in x and y for for particle using the Highlander formula.

    Args:
        thickness (float): thickness of the multiple scatterer in um
        energy (float): Energy of the traversing particle in MeV
        x_angle (float): Initial x angle of the particle before scattering
        y_angle (float): Initial y angle of the particle before scattering
        rad_length (float): Radiation length of each scatterer in um

    Returns:
        tuple[float, float]: _description_
    """
    var_angle = highlander_fast_electrons(thickness, rad_length, energy)
    return draw_2dgauss(x_angle, y_angle, var_angle, var_angle)


@njit
def fly(length, x_angle, y_angle):
    deltax = np.tan(x_angle) * length
    deltay = np.tan(y_angle) * length
    return deltax, deltay


def sample_dist_fast(
    pdf: list, *args: tuple, n: int, xmin: float, xmax: float, mode: str = "ntrue"
):
    """generates n values between xmin and xmax from the given distribution pdf using MC accept/reject
    expands on https://theoryandpractice.org/stats-ds-book/distributions/accept-reject.html

    Args:
        pdf (list): Sampling distribution
        args (tuple): optional arguments of the sampling distribution
        n (int): Number of draws
        xmin (float): minimum value of samples
        xmax (float): maximum value of samples
        mode (str, optional): Can be either 'ntrue' of 'pdftrue'. With 'ntrue': draws so long until number of accepts equals number of draws 'n'.
                            With 'pdftrue' draws only 'n' times. 'Defaults to "ntrue".

    Returns:
        list: Random draws from the given distribution
    """
    args = args[0]
    # fit for non normalized distributions
    x_vals = np.linspace(xmin, xmax, n)
    pdf_max = np.max(pdf(x_vals, *args))
    pdf_min = np.min(pdf(x_vals, *args))

    x = np.random.uniform(
        xmin, xmax, n
    )  # get uniform temporary x values between xmin and xmax
    y = np.random.uniform(
        pdf_min, pdf_max, n
    )  # get uniform random y values  between pdfmin and pdfmax

    if mode == "ntrue":
        return np.random.choice(x[y < pdf(x, *args)], size=n)

    elif mode == "pdftrue":
        new_n = int(n / (x[y < pdf(x, *args)].size / x.size))
        x = np.random.uniform(xmin, xmax, new_n)
        y = np.random.uniform(pdf_min, pdf_max, new_n)
        return x[y < pdf(x)]

    else:
        return x[y < pdf(x)]


def zeta(z: int, Z: int, A: int, beta: float, rho: float, x: float) -> float:
    """Helper function for the Landau distribution approximation of the Bethe Bloch formula

    Args:
        z (int): Charge of the incoming particle
        Z (int): Atomic number of the scatterer for the Landau distribution. Defaults to 14.
        A (int): Atomic mass number of the scatter. Defaults to 24.
        beta (float): beta value of the particle
        rho (float): Density of the scatterer
        x (float): thickness of scatterer

    Returns:
        float: Value of distribution at point x.
    """
    return 0.1535 * z**2 * Z * rho * x / (A * beta**2)


def lamb(zeta: float, delta: float, beta: float) -> float:
    """Variable of the Landau distribution

    Args:
        zeta (float): Approximatipon of the Bethe Bloch formula
        delta (float): Energy transfer during scattering
        beta (float): Beta parameter of particle

    Returns:
        float: Value for the Landau distribution
    """
    E = 0.000001
    return 1 / zeta * (delta - 0.025) - beta**2 - np.log(zeta / E) - 1 + np.e


def landau_approx(
    delta: list,
    x: float,
    z: float,
    Z: float,
    A: float,
    energy: float,
    rho: float,
    a: float,
):
    """Approximation of the Landau distribution from Behrens, S. E.; Melissinos, A.C. Univ. of Rochester Preprint UR-776 (1981)

    Args:
        delta (list): List containing energy transvers
        x (float): Thickness of the scatter
        z (float): charge of the incoming particle
        Z (float): Atomic number of the scatterering material
        A (float): Atomic mass number of the scattering material
        energy (float): Energy of the incoming particle
        rho (float): Density of the scattering material
        a (float): Amplitude of the Landau distribution

    Returns:
        list: Probability of energy loss
    """
    beta = np.sqrt(1 - 1 / (1 + (energy / 0.511) ** 2))
    zet = zeta(z, Z, A, beta, rho, x)
    coord = lamb(zet, delta, beta)
    return a * (1 / np.sqrt(2 * np.pi) * np.exp(-(coord + np.exp(-coord)) / 2))


def create_output_tracks(hit_table: tuple, folder: str) -> None:
    """Generate hdf5 file with the generated tracks. Warning this is very slow.

    Args:
        hit_table (tuple): Table of all particle tracks
        folder (str): Output path to save the tracks to
    """
    hits_descr = np.dtype(
        [
            ("energy", float),
            ("x_angle", float),
            ("y_angle", float),
            ("x", float),
            ("y", float),
            ("z", float),
            ("time_stamp", np.uint16),
        ]
    )

    numb = len(hit_table[0])
    hit_table_out = np.zeros(numb, dtype=hits_descr)
    hit_table_out = create_hit_array(hit_table, hit_table_out, numb)
    hit_table_out["time_stamp"] = hit_table[6]
    out_file_h5 = tb.open_file(filename=folder + "particle_tracks.h5", mode="w")
    output_hits_table = out_file_h5.create_table(
        where=out_file_h5.root,
        name="Tracks",
        description=hits_descr,
        title="Raw particle tracks",
        filters=tb.Filters(complib="blosc", complevel=5, fletcher32=False),
    )

    output_hits_table.append(hit_table_out)
    output_hits_table.flush()
    out_file_h5.close()


@njit
def create_hit_array(hit_table: tuple, hit_table_out: tb.table, numb: int) -> tb.table:
    """Sorts hit table in the output table.

    Args:
        hit_table (tuple): Input tuple of lists of hit parameters
        hit_table_out (tb.table): Output table prepared with all necessary  features.
        numb (int): Total number of particles

    Returns:
        tb.table: Sorted output table
    """
    for i in range(numb):
        hit_table_out["energy"][i] = hit_table[0][i]
        hit_table_out["x_angle"][i] = hit_table[1][i]
        hit_table_out["y_angle"][i] = hit_table[2][i]
        hit_table_out["x"][i] = hit_table[3][i]
        hit_table_out["y"][i] = hit_table[4][i]
        hit_table_out["z"][i] = hit_table[5][i]
    return hit_table_out
