#!/usr/bin/env python

import argparse
import copy
import sys
import warnings

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import pycwt

import nmfdkl


PNG = ".png"
WAV = ".wav"

def plot_wave(data, wave, power, scales, dt, ax):
    """
    from wavelets.py
    """
    axis = -1
    power = np.abs(wave.real)
    time = np.indices((data.shape[axis],)).squeeze() * dt

    Time, Scale = np.meshgrid(time, scales)
    ax.contourf(Time, Scale, power, 100)

    ax.set_yscale('log')
    ax.grid(True)

    ax.set_xlim(time.min(), time.max())

    return ax

def wave_power(wave):
    """
    from wavelets.py
    """
    # wavelet power spectrum
    power = np.abs(wave) ** 2
    return power

def scale_data(data, max):
    type = np.int16
    scale_factor = max / np.max(np.abs(data.real))
    scaled_data = data.real * scale_factor
    return scaled_data.astype(type)

def get_wt_nmfd(args):
    if args.wav is None:
        return
    fs, x = wavfile.read(args.wav[0])
    dt = 1.0/fs
    dj = 1.0/args.dj
    eps = np.spacing(1)
    do_nmfd = True
    mother = pycwt.Morlet(6.)
    wav_max = 2.0**15 - 1.0

    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(
        x, dt, dj=dj, wavelet=mother)

    # wavelet power spectrum
    power = wave_power(wave)
    #wt = copy.deepcopy(wa.wavelet_transform)
    fig, ax = plt.subplots()
    plot_wave(x, wave, power, scales, dt, ax)
    fig.savefig("".join((args.plot_prefix, PNG)))
    xi = pycwt.icwt(wave, scales, dt, dj=dj, wavelet=mother)
    wavfile.write("".join((args.wav_prefix, WAV)), fs, scale_data(xi, wav_max))
    nmfd = None
    if do_nmfd:
        nmfd = nmfdkl.NMFDKL(power, args.factors, args.templates)
        warnings.warn("nmfd")
        new_v = nmfd.nmfdkl(args.pre, args.post)
        warnings.warn("B")
        warnings.warn("B graph")
        denom = np.ones((new_v.shape[0], new_v.shape[1]))
        denom /= new_v[:, :, -1] + eps
        for r in xrange(new_v.shape[2] - 1):
            wa = copy.deepcopy(wave)
            wa.real *= new_v[:, :, r]*denom
            power = wave_power(wave)
            plot_wave(x, wa, power, scales, dt, ax)
            fig.savefig("{0}.{1:d}{2}".format(args.plot_prefix, r+1, PNG))
            xi = pycwt.icwt(wa, scales, dt, dj, mother)
            wavfile.write(
                "{0}.{1:d}{2}".format(args.wav_prefix, r+1, WAV), fs,
                scale_data(xi, wav_max))
                #np.savez(savez_path, W=W, H=H, cost=cost, V=wa.wavelet_transform, wt=wt,
                #new_v=new_v)

def test(wav_path, wav_prefix, plot_prefix):

    fs, x = wavfile.read(wav_path)
    dt = 1.0/fs
    dj = 1.0/8.0
    eps = np.spacing(1)
    mother = pycwt.Morlet(6.)
    wav_max = 2.0**15 - 1.0

    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(
        x, dt, dj=dj, wavelet=mother)

    # wavelet power spectrum
    power = wave_power(wave)
    nmfd = nmfdkl.NMFDKL(power, 4, 100)
    return (wave, scales, freqs, coi, fft, fftfreqs, power, nmfd)

def main():
    parser = argparse.ArgumentParser(
        description='Compute NMFD on CWT.')
    parser.add_argument(
        'wav', nargs=1, metavar="WAVFILE")
    parser.add_argument(
        '--wav-prefix', dest='wav_prefix') 
    parser.add_argument(
        '--plot-prefix', dest='plot_prefix')
    parser.add_argument(
        '--dj', type=int, dest='dj')
    parser.add_argument(
        '-t', '--templates', type=int, dest='templates')
    parser.add_argument(
        '-f', '--factors', type=int, dest='factors')
    parser.add_argument(
        '-p', '--pre', type=int, dest='pre')
    parser.add_argument(
        '-P', '--post', type=int, dest='post')
    args = parser.parse_args()
    get_wt_nmfd(args)

if __name__ == "__main__":
    main()
