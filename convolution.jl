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


function Gaussian(x:: StepRangeLen{Float64}, x0:: Float64, fwhm:: T) where {T<:Union{Float64, Int64}}
    sigma = fwhm/2sqrt(2log(2))
    height = 1/sigma*sqrt(2pi)
    Fx = height .* exp.( (-1 .* (x .- x0).^2)./ (2*sigma^2))
end

function Lorentzian(x:: StepRangeLen{Float64}, x0:: Float64, fwhm:: T) where {T<:Union{Float64, Int64}}
    gamma = fwhm/2
    Fx = 1 ./ ((pi * gamma) .* (1 .+ ((x .- x0)./ gamma).^2 ))
end























function convolve_signal(S:: Matrix{Float64}, fwhm:: T, time:: StepRangeLen{Float64}, Flag:: Int) where {T<:Union{Float64, Int64}} 

    nr:: Int64, nc:: Int64 = size(S)
    dt:: Float64 = sum(diff(time, dims=1))/(length(time)-1)
    nt:: Int64 = length(time)
    duration:: Float64 = fwhm*3 # must pad three time FHWM to deal with edges

    if dt != diff(time, dims=1)[1] ; error("Non-linear time range."); end
    if nr != nt && nc != nt; error("Inconsistent dimensions."); end
    if nc != nt; S = copy(transpose(S)); nr, nc = nc, nr; end # column always time axis

    tmin:: Float64 = minimum(time); tmax:: Float64 = maximum(time)
    tconv_min:: Float64 = tmin - duration ; tconv_max:: Float64 = tmax + duration
    tconv = tconv_min:dt:tconv_max # extended convoluted time vector
    ntc = length(tconv)

    padend:: Int64 = (duration/dt) # index at which 0 padding ends and signal starts
    padstart:: Int64 = (padend + tmax / dt) # index where signal ends and padding starts

    sC = zeros(Float64, nr, ntc)
    sC[1:nr, 1:padend] .= S[1:nr, 1] # extended and pad signal
    sC[1:nr, padend:padstart] .= S
    sC[1:nr, padstart:end] .= S[1:nr, end]

    conv = zeros(Float64, nr, ntc)
    
    for t = 1:ntc
        Flag == 0 ? K = Gaussian(tconv, tconv[t], fwhm) : K = Lorentzian(tconv, tconv[t], fwhm)
        for q = 1:nr
            #conv[q, t] = sum(sC[1:ntc, q] .* K) * dt
            for j = 1:ntc
                #sC[q, t] += sC[q, j] * K[j]
                conv[q, t] += sC[q, j] * K[j] * dt
            end
        end
    end
    return conv 
end

fwhm = 100.0
tvec = 0:0.5:1000
Flag = 0

#conv, tconv = convolve(signal, fwhm, tvec, Flag) 
