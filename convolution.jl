using MAT
#using LinearAlgebra
using SpecialMatrices
#using Plots

function Gaussian(x:: StepRangeLen{Float64}, x0:: Float64, fwhm:: T) where {T<:Union{Float64, Int64}}
    sigma = fwhm/2sqrt(2log(2))
    height = 1/sigma*sqrt(2pi)
    Fx = height .* exp.( (-1 .* (x .- x0).^2)./ (2*sigma^2))
end

function Lorentzian(x:: StepRangeLen{Float64}, x0:: Float64, fwhm:: T) where {T<:Union{Float64, Int64}}
    gamma = fwhm/2
    Fx = 1 ./ ((pi * gamma) .* (1 .+ ((x .- x0)./ gamma).^2 ))
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
        K = Gaussian(tconv, tconv[1], fwhm)
    elseif Flag == 1
        K = Lorentzian(tconv, tconv[1], fwhm)
    else
        error("Only Gaussian and Lorentzian functions currently available.")
    end
    
    K = Matrix(Circulant(K))

    return K
    end

    
function convolution_integral(signal:: Matrix{Float64}, kernel:: Matrix{Float64}, dt:: Float64)
    conv = signal * kernal'
    conv .*= dt
    return signal
end


function convolve(S:: Matrix{Float64}, K:: Matrix{Float64}, time:: StepRangeLen{Float64}, dt:: Float64, fwhm:: Float64) 

    nr:: Int64, nc:: Int64 = size(S)
    nt:: Int64 = length(time)
    if nr != nt && nc != nt; error("Inconsistent dimensions."); end
    if nc != nt; S = copy(transpose(S)); nr, nc = nc, nr; end # column always time axis
    tmin:: Float64 = minimum(time); tmax:: Float64 = maximum(time)
    duration = 3*fwhm

    padend:: Int64 = (duration/dt) # index at which 0 padding ends and signal starts
    padstart:: Int64 = (padend + tmax / dt) # index where signal ends and padding starts

    sC = zeros(Float64, nr, ntc)
    sC[1:nr, 1:padend] .= @view S[1:nr, 1] # extended and pad signal
    sC[1:nr, padend:padstart] .= S
    sC[1:nr, padstart:end] .= @view S[1:nr, end]

    conv = zeros(Float64, nr, ntc)
    conv = convolution_integral(sC, K, dt)
    
    return conv

    end

function run_conv()
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

    fwhm = 100.0
    tvec = 0:0.5:1000
    Flag = 0

kernel = generate_kernel(fwhm, tvec, Flag)
conv = convolve(signal, kernel, tvec, 0.5, fwhm)
end
