# Neo Vague Denoiser (forked from VapourSynth-VagueDenoiser)

Neo Vague Denoiser Copyright(C) 2020 Xinyue Lu, and previous developers

Vague Denoiser is a wavelet based denoiser.

Basically, it transforms each frame from the video input into the wavelet domain, using Cohen-Daubechies-Feauveau 9/7. Then it applies some filtering to the obtained coefficients. It does an inverse wavelet transform after. Due to wavelet properties, it should give a nice smoothed result, and reduced noise, without blurring picture features.

It was originally written by Lefungus, and later modified by Kurosu and Fizick for further improvement. VapourSynth-VagueDenoiser was ported to VapourSynth interface and refactored by HolyWu. Kudos to them for creating and improving this fantastic tool.

This project backports VapourSynth-VagueDenoiser to AviSynth+. Parameter names follow VapourSynth-VagueDenoiser.

## Usage

```python
# AviSynth+
LoadPlugin("neo-vague-denoiser.dll")
neo_vd(clip, threshold=2.0, nsteps=6, y=3, u=3, v=3, ...)
# VapourSynth
core.neo_vd.VagueDenoiser(clip, threshold=2.0, nsteps=6, planes=[0,1,2], ...)
```

Parameters:

[Check original VapourSynth-VagueDenoiser usage documents.](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VagueDenoiser/blob/master/README.md)

- *y*, *u*, *v* (AviSynth+ only)

    Whether a plane is to be filtered.

        1 - Do not touch, leaving garbage data
        2 - Copy from origin
        3 - Process

    Default: 3.

## License

* GPLv2.
