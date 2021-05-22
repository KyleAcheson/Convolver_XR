using MAT
#using LinearAlgebra
using SpecialMatrices
#using Plots

wd = "/Users/kyleacheson/MATLAB/SCATTERING/ROT_AVG/CS2_UED_Corrected"

inputfile = matopen(wd*"/AllTraj_Signal.mat")
signal = read(inputfile, "pdW")
close(inputfile)
replace!(signal, NaN=>0)
signal = signal[2:671, 1:2001]
signal[:, 1] = signal[:, 2]

inputfile = matopen(wd*"/Vectors.mat")
qAng = read(inputfile, "qAng")
tvec = read(inputfile, "tt")


struct Gaussian
    Fx:: AbstractVector{Float64}
    sigma:: Float64
    height:: Float64
    fwhm:: Float64
    function Gaussian(fwhm:: Float64, x:: StepRangeLen{Float64}, x0:: Float64)
        sigma = fwhm/2sqrt(2log(2))
        height = 1/sigma*sqrt(2pi)
        Fx = height .* exp.( (-1 .* (x .- x0).^2)./ (2*sigma^2))
        new(Fx, sigma, height, fwhm)
    end
end

struct Lorentzian 
    Fx:: AbstractVector{Float64}
    gamma:: Float64
    fwhm:: Float64
    function Lorentzian(fwhm:: Float64, x:: StepRangeLen{Float64}, x0:: Float64)
        gamma = fwhm/2
        Fx = 1 ./ ((pi * gamma) .* (1 .+ ((x .- x0)./ gamma).^2 ))
        new(Fx, gamma, fwhm)
    end
end


"""
# convole
`convolve(S:: AbstractArray, fwhm:: Real, time:: AbstractArray, type:: Int)`:

Convolves a scattering signal with a gaussian or lorentzian over the time axis.
The same convolution is applied to every row of the input signal. 

`S:: AbstractArray` = Input signal

`fwhm:: Real` = Combined FWHM of pump and probe pulses to be convolved with

`time:: AbstractArray` = Time vector, will be extended by 3*FWHM

`type:: Int` = 0 for Gaussian, 1 for Lorentzian

FWHM is converted to σ or γ depending on choice of function.

Input signal will be padded by 3*FWHM before t=0, setting t<0 to the signal at t=0.
In addition the signal is extended by the same amount after t=end, setting t>end
to the signal at t=end.
"""
function convolve(S:: Array{Float64}, fwhm:: Float64, time:: Array{Float64}, type:: Int)

    nr:: Int64, nc:: Int64 = size(S)
    nt:: Int64 = length(time)
    dt:: Float64 = sum(diff(time, dims=2))/(length(time)-1)
    duration:: Float64 = fwhm*3 # must pad three time FHWM to deal with edges

    if dt != diff(time, dims=2)[1] ; error("Non-linear time range."); end
    if nr != nt && nc != nt; error("Inconsistent dimensions."); end
    if nc != nt; S = transpose(S); nr, nc = nc, nr; end # column always time axis

    tmin:: Float64 = minimum(time); tmax:: Float64 = maximum(time)
    tconv_min:: Float64 = tmin - duration ; tconv_max:: Float64 = tmax + duration
    tconv = tconv_min:dt:tconv_max # extended convoluted time vector
    ntc:: Int64 = length(tconv) 
    padend:: Int64 = (duration/dt) # index at which 0 padding ends and signal starts
    padstart:: Int64 = (padend + tmax / dt) # index where signal ends and padding starts

    type == 0 ? Kv = Gaussian(fwhm, tconv, tconv[1]) : Kv = Lorentzian(fwhm, tconv, tconv[1])
    println("Convolving signal with a $(typeof(Kv)) function with FWHM: $(Kv.fwhm) fs.")

    K:: Array{Float64} = Circulant(Kv.Fx)
    sC = zeros(Float64, nr, ntc)
    sC[1:nr, 1:padend] .= S[1:nr, 1] # extended and pad signal
    sC[1:nr, padend:padstart] .= S
    sC[1:nr, padstart:end] .= S[1:nr, end]

    println("""
            Signal padded by 3*FWHM ( $(duration) fs ) forwards and backwards.
            Original time length: $(nt), Extended time: $(ntc), Diff: $(ntc-nt) steps.
            Padded signal size: $(size(sC)).
            Kernel size: $(size(K[1, :])).
            """)
    conv = zeros(Float64, nr, ntc)
    for t in 1:ntc
        for q in 1:nr
            #conv[q, t] = sum(sC[q, 1:ntc] .* K[t, :])*dt
            conv[q, t] = 0.0
            for j in 1:ntc
                conv[q, t] += sC[q, j] * K[t, j] * dt
            end
        end
    end

    #conv = mapslices(x -> K*x, sC; dims=2)
    return conv, tconv
end

fwhm = 100.0
conv, tconv = convolve(signal, fwhm, tvec, 0)
