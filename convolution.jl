using MAT
using SpecialMatrices


struct Gaussian
    Fx:: Circulant{Float64}
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
        Fx = Circulant(Fx)
        tconv = x; ntc = length(x)
        dt = tconv[2]-tconv[1]
        nt = nt
        new(Fx, sigma, height, fwhm, tconv, ntc, duration, dt, tmin, tmax, nt)
    end
end


struct Lorentzian 
    Fx:: Circulant{Float64}
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
        Fx = Circulant(Fx)
        tconv = x; ntc = length(x)
        dt = tconv[2]-tconv[1]
        nt = nt
        new(Fx, gamma, fwhm, tconv, ntc, duration, dt, tmin, tmax, nt)
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
    
@views function convolution_integral(conv:: Matrix{Float64}, ntc:: Int64, nr:: Int64, signal:: Matrix{Float64}, kernel:: Circulant{Float64}, dt:: Float64)
    for t = 1:ntc
        for q = 1:nr
            for j=1:ntc
                conv[q, t] += signal[q, t] * kernel[t, j] * dt
            end
        end
    end
    return conv
end


function convolve(S:: Matrix{Float64}, K:: T) where {T<:Union{Gaussian, Lorentzian}}

    nr:: Int64, nc:: Int64 = size(S)
    if nr != K.nt && nc != K.nt; error("Inconsistent dimensions."); end
    if nc != K.nt; S = copy(transpose(S)); nr, nc = nc, nr; end # column always time axis

    padend:: Int64 = (K.duration/K.dt) # index at which 0 padding ends and signal starts
    padstart:: Int64 = (padend + K.tmax / K.dt) # index where signal ends and padding starts

    println("Convolving signal with a $(typeof(K)) function with FWHM: $(K.fwhm) fs.")

    sC = zeros(Float64, nr, K.ntc)
    sC[1:nr, 1:padend] .= @view S[1:nr, 1] # extended and pad signal
    sC[1:nr, padend:padstart] .= S
    sC[1:nr, padstart:end] .= @view S[1:nr, end]
    conv = zeros(Float64, nr, K.ntc)

    @time conv = convolution_integral(conv, K.ntc, nr, sC, K.Fx, K.dt)
    
    #for t in 1:K.ntc
    #    for q in 1:nr
    #        #conv[q, t] = sum(collect(sC[q, 1:K.ntc]) .* collect(K.Fx[t, :]))*K.dt
    #        for j in 1:K.ntc
    #            conv[q, t] += sC[q, j] * K.Fx[t, j] * K.dt
    #        end
    #    end
    #end

    return conv 
end

function conv_wrapper()

    wd = "/Users/kyleacheson/MATLAB/SCATTERING/ROT_AVG/CS2_UED_Corrected"
    inputfile = matopen(wd*"/AllTraj_Signal.mat")
    signal = read(inputfile, "pdW")
    close(inputfile)
    replace!(signal, NaN=>0)
    signal = signal[2:671, 1:2001]
    signal[:, 1] = signal[:, 2]

    inputfile = matopen(wd*"/Vectors.mat")
    qAng = read(inputfile, "qAng")
    #const tvec = read(inputfile, "tt")

    fwhm = 100.0
    tvec = 0:0.5:1000
    Flag = 0
    kernel = generate_kernel(fwhm, tvec, Flag)
    conv, tconv = convolve(signal, kernel) 

end
