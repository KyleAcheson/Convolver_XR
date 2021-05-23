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


mutable struct Gaussian
    Fx:: Array{Float64}
    sigma:: Float64
    height:: Float64
    fwhm:: Float64
    tconv:: StepRangeLen{Float64}
    ntc:: Int64
    duration:: Float64
    dt:: Float64
    tmin :: Float64
    tmax:: Float64
    nt:: Int64
    function Gaussian(fwhm:: Float64, x:: StepRangeLen{Float64}, x0:: Float64, tmin:: Float64, tmax::Float64, nt:: Int64)
        duration = fwhm*3
        sigma = fwhm/2sqrt(2log(2))
        height = 1/sigma*sqrt(2pi)
        Fx = height .* exp.( (-1 .* (x .- x0).^2)./ (2*sigma^2))
        tconv = x; ntc = length(x)
        dt = tconv[2]-tconv[1]
        nt = nt
        new(Fx, sigma, height, fwhm, tconv, ntc, duration, dt, tmin, tmax, nt)
    end
    function Gaussian(G:: Gaussian, x0:: Float64)
        Fx = G.height .* exp.( (-1 .* (G.tconv .- x0).^2)./ (2*G.sigma^2))
        new(Fx)
    end
end


mutable struct Lorentzian 
    Fx:: Array{Float64}
    gamma:: Float64
    fwhm:: Float64
    tconv:: StepRangeLen{Float64}
    ntc:: Int64
    duration:: Float64
    dt:: Float64
    tmin :: Float64
    tmax:: Float64
    nt:: Int64
    function Lorentzian(fwhm:: Float64, x:: StepRangeLen{Float64}, x0:: Float64, tmin::Float64, tmax:: Float64, nt:: Int64)
        gamma = fwhm/2
        duration = fwhm*3
        Fx = 1 ./ ((pi * gamma) .* (1 .+ ((x .- x0)./ gamma).^2 ))
        tconv = x; ntc = length(x)
        dt = tconv[2]-tconv[1]
        nt = nt
        new(Fx, gamma, fwhm, tconv, ntc, duration, dt, tmin, tmax, nt)
    end
    function Lorentzian(L:: Lorentzian, x0:: Float64)
        Fx = 1 ./ ((pi * L.gamma) .* (1 .+ ((L.tconv .- x0)./ L.gamma).^2 ))
        new(Fx)
    end
end


function generate_kernel(fwhm:: Float64, time:: StepRangeLen{Float64}, Flag:: Int)

    dt:: Float64 = sum(diff(time, dims=1))/(length(time)-1)
    nt:: Int64 = length(time)
    duration:: Float64 = fwhm*3 # must pad three time FHWM to deal with edges

    if dt != diff(time, dims=1)[1] ; error("Non-linear time range."); end

    tmin:: Float64 = minimum(time); tmax:: Float64 = maximum(time)
    tconv_min:: Float64 = tmin - duration ; tconv_max:: Float64 = tmax + duration
    tconv = tconv_min:dt:tconv_max # extended convoluted time vector

    if Flag == 0
        K = Gaussian(fwhm, tconv, tconv[1], tmin, tmax, nt)
    elseif Flag == 1
        K = Lorentzian(fwhm, tconv, tconv[1], tmin, tmax, nt)
    else
        error("Only Gaussian and Lorentzian functions currently available.")
    end

    return K

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

function convolve(S:: Array{Float64}, K:: T) where {T<:Union{Gaussian, Lorentzian}}

    nr:: Int64, nc:: Int64 = size(S)

    if nr != K.nt && nc != K.nt; error("Inconsistent dimensions."); end
    if nc != K.nt; S = transpose(S); nr, nc = nc, nr; end # column always time axis

    padend:: Int64 = (K.duration/K.dt) # index at which 0 padding ends and signal starts
    padstart:: Int64 = (padend + K.tmax / K.dt) # index where signal ends and padding starts

    println("Convolving signal with a $(typeof(K)) function with FWHM: $(K.fwhm) fs.")

    sC = zeros(Float64, nr, K.ntc)
    sC[1:nr, 1:padend] .= S[1:nr, 1] # extended and pad signal
    sC[1:nr, padend:padstart] .= S
    sC[1:nr, padstart:end] .= S[1:nr, end]

    #println("""
    #        Signal padded by 3*FWHM ( $(duration) fs ) forwards and backwards.
    #        Original time length: $(nt), Extended time: $(ntc), Diff: $(ntc-nt) steps.
    #        Padded signal size: $(size(sC)).
    #        Kernel size: $(size(K[1, :])).
    #        """)
    conv = zeros(Float64, nr, K.ntc)
    #a = 1.0
    #dt = 0.5
    
    for t in 1:K.ntc
        Gaussian(K, K.tconv[t])
        println(t)
        for q in 1:nr
            #conv[q, t] = sum(collect(sC[q, 1:K.ntc]) .* collect(K.Fx[t, :]))*K.dt
            conv[q, t] = sum(sC[q, 1:K.ntc] .* K.Fx) * K.dt
            #for j in 1:K.ntc
                #a += 1.0 * K.dt
                #sC[q, t] += sC[q, j] * K.Fx[t, j]
                #@time conv[q, t] += sC[q, j] * K.dt
            #end
        end
    end

    #conv = mapslices(x -> K*x, sC; dims=2)
    return conv 
end

fwhm = 100.0
tvec = 0:0.5:1000
kernel = generate_kernel(fwhm, tvec, 0)
#conv, tconv = convolve(signal, kernel) 
