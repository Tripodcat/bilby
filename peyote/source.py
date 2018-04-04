from __future__ import division, print_function
import peyote
import numpy as np
import os.path
from astropy.table import Table
from peyote.utils import sampling_frequency, nfft

try:
    import lal
    import lalsimulation as lalsim
except ImportError:
    print("lal is not installed")


class Source:
    def __init__(self, name, sampling_frequency, time_duration):
        self.name = name
        self.sampling_frequency = sampling_frequency
        self.time_duration = time_duration
        self.time = peyote.utils.create_time_series(
            sampling_frequency, time_duration)
        self.nsamples = len(self.time)
        self.ff = np.fft.rfftfreq(self.nsamples, 1/self.sampling_frequency)


class SimpleSinusoidSource(Source):
    """ A simple example of a sinusoid source

    model takes one parameter `parameters`, a dictionary of Parameters and
    returns the waveform model.

    """

    parameter_keys = ['A', 'f']

    def time_domain_strain(self, parameters):
        return {'plus': parameters['A'] * np.sin(2 * np.pi * parameters['f'] * self.time),
                'cross': parameters['A'] * np.cos(2 * np.pi * parameters['f'] * self.time)}

    def frequency_domain_strain(self, parameters):
        hf = {}
        ht = self.time_domain_strain(parameters)
        for mode in ht.keys():
            hf[mode], _ = peyote.utils.nfft(ht[mode], self.sampling_frequency)
        return hf


class BinaryBlackHole(Source):
    """
    A way of getting a BBH waveform from lal
    """
    def waveform_from_lal(self, frequencies, parameters):
        luminosity_distance = parameters['luminosity_distance'] * 1e6 * lal.PC_SI
        mass_1 = parameters['mass_1'] * lal.MSUN_SI
        mass_2 = parameters['mass_2'] * lal.MSUN_SI

        longitude_ascending_nodes = 0.0
        eccentricity = 0.0
        meanPerAno = 0.0

        waveform_dictionary = lal.CreateDict()

        approximant = lalsim.GetApproximantFromString(parameters['waveform_approximant'])

        # delta_f = frequencies[1]-frequencies[0]
        # minimum_frequency = np.min(frequencies)
        # print(minimum_frequency)
        # maximum_frequency = np.max(frequencies)

        hplus,hcross = lalsim.SimInspiralChooseFDWaveform(mass_1, mass_2,
                        parameters['spin_1'][0], parameters['spin_1'][1], parameters['spin_1'][2],
                        parameters['spin_2'][0], parameters['spin_2'][1], parameters['spin_2'][2],
                        luminosity_distance, parameters['inclination_angle'],
                        parameters['waveform_phase'],
                        longitude_ascending_nodes, eccentricity, meanPerAno,
                        parameters['deltaF'], parameters['fmin'], parameters['fmax'],
                        parameters['reference_frequency'],
                        waveform_dictionary, approximant)

        h_plus = hplus.data.data
        h_cross = hcross.data.data

        return h_plus, h_cross


class Glitch(Source):
    def __init__(self, name):
        Source.__init__(self, name)


class AstrophysicalSource(Source):

    def __init__(self, name, right_ascension, declination, luminosity_distance):
        Source.__init__(self, name)
        self.right_ascension = right_ascension
        self.declination = declination
        self.luminosity_distance = luminosity_distance


class CompactBinaryCoalescence(AstrophysicalSource):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle, eccentricity):
        AstrophysicalSource.__init__(self, name, right_ascension, declination, luminosity_distance)
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.spin_1 = spin_1
        self.spin_2 = spin_2
        self.coalescence_time = coalescence_time  # tc
        self.inclination_angle = inclination_angle  # iota
        self.waveform_phase = waveform_phase  # phi
        self.polarisation_angle = polarisation_angle  # psi
        self.eccentricity = eccentricity


class Supernova(AstrophysicalSource):
    def __init__(self, name, right_ascension, declination, luminosity_distance):
        AstrophysicalSource.__init__(self, name, right_ascension, declination, luminosity_distance)


# class BinaryBlackHole(CompactBinaryCoalescence):
#     def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
#                  coalescence_time, inclination_angle, waveform_phase, polarisation_angle, eccentricity):
#         CompactBinaryCoalescence.__init__(self, name, right_ascension, declination, luminosity_distance, mass_1,
#                                           mass_2, spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
#                                           polarisation_angle, eccentricity)



class BinaryNeutronStar(CompactBinaryCoalescence):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle, eccentricity,
                 tidal_deformability_1, tidal_deformability_2):
        CompactBinaryCoalescence.__init__(self, name, right_ascension, declination, luminosity_distance, mass_1,
                                          mass_2, spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
                                          polarisation_angle, eccentricity)
        self.tidal_deformability_1 = tidal_deformability_1  # lambda parameter for Neutron Star 1
        self.tidal_deformability_2 = tidal_deformability_2  # lambda parameter for Neutron Star 2


class NeutronStarBlackHole(CompactBinaryCoalescence):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle, eccentricity,
                 tidal_deformability):
        CompactBinaryCoalescence.__init__(self, name, right_ascension, declination, luminosity_distance, mass_1,
                                          mass_2, spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
                                          polarisation_angle, eccentricity)
        self.tidal_deformability = tidal_deformability  # lambda parameter for Neutron Star


class BinaryNeutronStarMergerNumericalRelativity(Source):
    """ Loads in NR simulations of BNS merger

    takes parameters mean_mass, mass_ratio and equation_of_state, directory_path

    returns time,hplus,hcross,freq,Hplus(freq),Hcross(freq)

    """

    def model(self, parameters):
        mean_mass_string = '{:.0f}'.format(parameters['mean_mass'] * 1000)
        eos_string = parameters['equation_of_state']
        mass_ratio_string = '{:.0f}'.format(parameters['mass_ratio'] * 10)
        directory_path = parameters['directory_path']

        file_name = '{}-q{}-M{}.csv'.format(eos_string, mass_ratio_string, mean_mass_string)
        full_filename = '{}/{}'.format(directory_path, file_name)

        if not os.path.isfile(full_filename):
            print('{} does not exist'.format(full_filename)) # add exception
            return(-1)
        else: # ok file exists
            strain_table = Table.read(full_filename)
            Hplus, _ = nfft(strain_table["hplus"], sampling_frequency(strain_table['time']))
            Hcross, frequency = nfft(strain_table["hcross"], sampling_frequency(strain_table['time']))
            return(strain_table['time'],strain_table["hplus"],strain_table["hcross"],frequency,Hplus,Hcross)
