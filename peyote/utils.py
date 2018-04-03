import numpy as np
from astropy.time import Time

# Constants
speed_of_light = 299792458.0  # speed of light in m/s


def sampling_frequency(time_series):
    """
    Calculate sampling frequency from a time series
    """
    tol = 1e-10
    if np.ptp(np.diff(time_series)) > tol:
        raise ValueError("Your time series was not evenly sampled")
    else:
        return 1. / (time_series[1] - time_series[0])


def create_time_series(sampling_frequency, duration, starting_time = 0.):
    return np.arange(starting_time, duration, 1./sampling_frequency)


def ra_dec_to_theta_phi(ra, dec, gmst):
    """
    Convert from RA and DEC to polar coordinates on celestial sphere
    Input:
    ra - right ascension in radians
    dec - declination in radians
    gmst - Greenwich mean sidereal time of arrival of the signal in radians
    Output:
    theta - zenith angle in radians
    phi - azimuthal angle in radians
    """
    phi = ra - gmst
    theta = np.pi / 2 - dec
    return theta, phi


def gps_time_to_gmst(time):
    """
    Convert gps time to Greenwich mean sidereal time in radians
    Input:
    time - gps time
    Output:
    gmst - Greenwich mean sidereal time in radians
    """
    gps_time = Time(time, format='gps', scale='utc')
    gmst = gps_time.sidereal_time('mean', 'greenwich').value * np.pi / 12
    return gmst


def create_white_noise(sampling_frequency, duration):
    """
    Create white_noise which is then coloured by a given PSD
    """

    number_of_samples = duration * sampling_frequency
    number_of_samples = int(np.round(number_of_samples))

    # prepare for FFT
    number_of_frequencies = (number_of_samples-1)//2
    delta_freq = 1./duration

    f = delta_freq * np.linspace(1, number_of_frequencies, number_of_frequencies)

    norm1 = 0.5*(1./delta_freq)**0.5
    re1 = np.random.normal(0, norm1, int(number_of_frequencies))
    im1 = np.random.normal(0, norm1, int(number_of_frequencies))
    z1 = re1 + 1j*im1


    # freq domain solution for htilde1, htilde2 in terms of z1, z2
    htilde1 = z1
    # convolve data with instrument transfer function
    otilde1 = htilde1 * 1.
    # set DC and Nyquist = 0
    # python: we are working entirely with positive frequencies
    if np.mod(number_of_samples, 2) == 0:
        otilde1 = np.concatenate(([0], otilde1, [0]))
        f = np.concatenate(([0], f, [sampling_frequency / 2.]))
    else:
        # no Nyquist frequency when N=odd
        otilde1 = np.concatenate(([0], otilde1))
        f = np.concatenate(([0], f))

    # normalise for positive frequencies and units of strain/rHz
    white_noise = otilde1
    # python: transpose for use with infft
    white_noise = np.transpose(white_noise)
    f = np.transpose(f)

    return white_noise, f


def nfft(ht, Fs):
    '''
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)

    ht = time series
    Fs = sampling frequency

    returns
    hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f = frequencies associated with hf
    '''
    # add one zero padding if time series does not have even number of sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    LL = len(ht)
    # frequency range
    ff = Fs / 2 * np.linspace(0, 1, LL/2+1)

    # calculate FFT
    # rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of strain / sqrt(Hz)
    hf = hf / Fs

    return hf, ff


def infft(hf, Fs):
    '''
    inverse FFT for use in conjunction with nfft
    eric.thrane@ligo.org
    input:
    hf = single-side FFT calculated by fft_eht
    Fs = sampling frequency
    output:
    h = time series
    '''
    # use irfft to work with positive frequencies only
    h = np.fft.irfft(hf)
    # undo LAL/Lasky normalisation
    h = h*Fs

    return h


def time_delay_geocentric(detector1, detector2, ra, dec, time):
    '''
    Calculate time delay between two detectors in geocentric coordinates based on XLALArrivaTimeDiff in TimeDelay.c
    Input:
    detector1 - cartesian coordinate vector for the first detector in the geocentric frame
                generated by the Interferometer class as self.vertex
    detector2 - cartesian coordinate vector for the second detector in the geocentric frame
    To get time delay from Earth center, use detector2 = np.array([0,0,0])
    ra - right ascension of the source in radians
    dec - declination of the source in radians
    time - GPS time in the geocentric frame
    Output:
    delta_t - time delay between the two detectors in the geocentric frame
    '''
    gmst = gps_time_to_gmst(time)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    omega = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    delta_D = detector2 - detector1
    delta_t = np.dot(omega, delta_D) / speed_of_light
    return delta_t


def get_polarization_tensor(ra, dec, time, psi, mode):
    """
    Calculate the polarization tensor for a given sky location and time

    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note: there is a typo in the definition of the wave-frame in Nishizawa et al.

    :param ra: right ascension in radians
    :param dec: declination in radians
    :param time: geocentric GPS time
    :param psi: binary polarisation angle counter-clockwise about the direction of propagation
    :param mode: polarisation mode
    :return: polarization_tensor(ra, dec, time, psi, mode): polarization tensor for the specified mode.
    """
    greenwich_mean_sidereal_time = gps_time_to_gmst(time)
    theta, phi = ra_dec_to_theta_phi(ra, dec, greenwich_mean_sidereal_time)
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)
    omega = np.cross(m, n)

    if mode == "plus":
        # polarization_tensor = np.einsum('i,j->ij', m, m) - np.einsum('i,j->ij', n, n)
        polarization_tensor = np.array([[m[0]*m[0]-n[0]*n[0], m[0]*m[1]-n[0]*n[1], m[0]*m[2]-n[0]*n[2]],
                                       [m[1]*m[0]-n[1]*n[0], m[1]*m[1]-n[1]*n[1], m[1]*m[2]-n[1]*n[2]],
                                       [m[2]*m[0]-n[2]*n[0], m[2]*m[1]-n[2]*n[1], m[2]*m[2]-n[2]*n[2]]])
    elif mode == "cross":
        polarization_tensor = np.einsum('i,j->ij', m, n) + np.einsum('i,j->ij', n, m)
    elif mode == "breathing":
        polarization_tensor = np.einsum('i,j->ij', m, m) + np.einsum('i,j->ij', n, n)
    elif mode == "longitudinal":
        polarization_tensor = np.sqrt(2) * np.einsum('i,j->ij', omega, omega)
    elif mode == "x":
        polarization_tensor = np.einsum('i,j->ij', m, omega) + np.einsum('i,j->ij', omega, m)
    elif mode == "y":
        polarization_tensor = np.einsum('i,j->ij', n, omega) + np.einsum('i,j->ij', omega, n)
    else:
        print("Not a polarization mode!")
        return None
    return polarization_tensor


def get_vertex_position_geocentric(latitude, longitude, elevation):
    """
    Calculate the position of the IFO vertex in geocentric coordiantes in meters.

    Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
    See Section 2.1 of LIGO-T980044-10 for the correct expression
    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m
    radius = semi_major_axis**2 * (semi_major_axis**2 * np.cos(latitude)**2
                                   + semi_minor_axis**2 * np.sin(latitude)**2)**(-0.5)
    x_comp = (radius + elevation) * np.cos(latitude) * np.cos(longitude)
    y_comp = (radius + elevation) * np.cos(latitude) * np.sin(longitude)
    z_comp = ((semi_minor_axis / semi_major_axis)**2 * radius + elevation) * np.sin(latitude)
    return np.array([x_comp, y_comp, z_comp])
