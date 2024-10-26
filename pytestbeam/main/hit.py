import numpy as np
from numba import njit, prange
from numba.typed import List
import pylandau
from numba_progress import ProgressBar
import tables as tb


def tracks(beam, devices, materials, log):
    log.info("Creating telescope setup")
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

    Z = List()
    [Z.append(material["Z"]) for material in materials]
    A = List()
    [A.append(material["A"]) for material in materials]
    rho = List()
    [rho.append(material["rho"]) for material in materials]
    rad_length = List()
    [rad_length.append(material["rad_length"]) for material in materials]

    x_extend_pos = List()
    [
        x_extend_pos.append(dut["column_pitch"] * dut["column"] / 2 + dut["delta_x"])
        for dut in devices
    ]
    x_extend_neg = List()
    [
        x_extend_neg.append(-dut["column_pitch"] * dut["column"] / 2 + dut["delta_x"])
        for dut in devices
    ]
    y_extend_pos = List()
    [
        y_extend_pos.append(dut["row_pitch"] * dut["row"] / 2 + dut["delta_y"])
        for dut in devices
    ]
    y_extend_neg = List()
    [
        y_extend_neg.append(-dut["row_pitch"] * dut["row"] / 2 + dut["delta_y"])
        for dut in devices
    ]
    x = List()
    [x.append(i) for i in [np.random.normal(beam["loc_x"], beam["sigma_x"])]]
    y = List()
    [y.append(i) for i in [np.random.normal(beam["loc_y"], beam["sigma_y"])]]
    z = List()
    [z.append(i) for i in [0]]
    numb_events = beam["nmb_particles"]
    device_nmb = len(devices)
    z_positions = List()
    [z_positions.append(i) for i in [dut["z_position"] for dut in devices]]
    scatter_thickness = List()
    [scatter_thickness.append(i) for i in [dut["thickness"] for dut in devices]]
    time_stamp = List()
    [time_stamp.append(i) for i in [np.random.poisson(particle_distance)]]
    angle_x, angle_y = scatter(
        scatter_thickness[0], beam["energy"], 0, 0, rad_length[0]
    )
    anglex = List()
    [anglex.append(i) for i in [beam["x_angle"] + angle_x]]
    angley = List()
    [angley.append(i) for i in [beam["y_angle"] + angle_y]]

    log.info("Generating particle tracks")
    energy_lost = np.zeros((device_nmb, numb_events))
    for dev in range(device_nmb):
        energy_lost[dev] = sample_landau_dist_fast(
            landau,
            numb_events,
            0,
            0.5,
            energy=(beam["energy"] - np.mean(energy_lost[dev - 1])),
            z=-1,
            Z=Z[dev],
            A=A[dev],
            rho=rho[dev],
            d=scatter_thickness[dev] * 10 ** (-4),
            mode="ntrue",
        )

    energy = List()
    [energy.append(i) for i in [beam["energy"] - energy_lost[0][0]]]

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
    progress_proxy,
):
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
def draw_2dgauss(locx, locy, sigx, sigy):
    return np.random.normal(locx, sigx), np.random.normal(locy, sigy)


@njit
def draw_1dgauss(loc, sig):
    return np.random.normal(loc, sig)


@njit
def highlander_fast_electrons(thickness, rad_length, energy):
    z = -1
    # p = np.sqrt(energy**2 - 0.511**2)
    p = energy
    beta = np.sqrt(1 - 1 / (1 + (energy / 0.511) ** 2))
    epsilon = thickness / rad_length
    return np.sqrt(
        (13.6 / (beta * p) * z) ** 2 * epsilon * (1 + 0.038 * np.log(epsilon)) ** 2
    )


@njit
def scatter(thickness, energy, x_angle, y_angle, rad_length):
    var_angle = highlander_fast_electrons(thickness, rad_length, energy)
    return draw_2dgauss(x_angle, y_angle, var_angle, var_angle)


@njit
def fly(length, x_angle, y_angle):
    deltax = np.tan(x_angle) * length
    deltay = np.tan(y_angle) * length
    return deltax, deltay


def sample_landau_dist_fast(
    pdf, n, xmin, xmax, energy=30, z=-1, Z=14, A=24, rho=2.33, d=0.02, mode="ntrue"
):
    """generates n values between xmin and xmax from the given distribution pdf using MC accept/reject
    expands on https://theoryandpractice.org/stats-ds-book/distributions/accept-reject.html
    """

    # fi for non normalized distributions
    x_vals = np.linspace(xmin, xmax, n)
    pdf_max = np.max(pdf(x_vals, energy, z, Z, A, rho, d))
    pdf_min = np.min(pdf(x_vals, energy, z, Z, A, rho, d))

    x = np.random.uniform(
        xmin, xmax, n
    )  # get uniform temporary x values between xmin and xmax
    y = np.random.uniform(
        pdf_min, pdf_max, n
    )  # get uniform random y values  between pdfmin and pdfmax

    if mode == "ntrue":
        return np.random.choice(x[y < pdf(x, energy, z, Z, A, rho, d)], size=n)

    elif mode == "pdftrue":
        new_n = int(n / (x[y < pdf(x)].size / x.size))
        x = np.random.uniform(xmin, xmax, new_n)
        y = np.random.uniform(pdf_min, pdf_max, new_n)
        return x[y < pdf(x)]

    else:
        return x[y < pdf(x)]


def zeta(z, Z, A, beta, rho, x):
    return 0.1535 * z**2 * Z * rho * x / (A * beta**2)


def lamb(zeta, delta, beta):
    E = 0.000001
    return 1 / zeta * (delta - 0.025) - beta**2 - np.log(zeta / E) - 1 + np.e


def landau(delta, energy, z, Z, A, rho, x):
    beta = np.sqrt(1 - 1 / (1 + (energy / 0.511) ** 2))
    zet = zeta(z, Z, A, beta, rho, x)
    return pylandau.landau(lamb(zet, delta, beta)) / zet


def create_output_tracks(hit_table, folder):
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
def create_hit_array(hit_table, hit_table_out, numb):
    for i in range(numb):
        hit_table_out["energy"][i] = hit_table[0][i]
        hit_table_out["x_angle"][i] = hit_table[1][i]
        hit_table_out["y_angle"][i] = hit_table[2][i]
        hit_table_out["x"][i] = hit_table[3][i]
        hit_table_out["y"][i] = hit_table[4][i]
        hit_table_out["z"][i] = hit_table[5][i]
    return hit_table_out
