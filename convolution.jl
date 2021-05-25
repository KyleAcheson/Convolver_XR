using MAT
using LinearAlgebra
using SpecialMatrices

function run_conv(savemat:: Bool)
    wd = "/home/kyle/Convolver_XR"
    inputfile = matopen(wd*"/AllTraj_Signal.mat")
    signal = read(inputfile, "pdW")
    close(inputfile)
    replace!(signal, NaN=>0)
    signal = signal[2:671, 1:2001]
    signal[:, 1] = signal[:, 2]

    inputfile = matopen(wd*"/Vectors.mat")
    qAng = read(inputfile, "qAng")
    qAng = qAng[2:end]
    tvec = read(inputfile, "tt")

    fwhm = 150.0
    tvec = 0:0.5:1000
    Flag = 0
    conv = convolve(signal, tvec, fwhm, Flag)
    
    #tconv = collect(tconv)

    if savemat
        file = matopen("convolution.mat", "w")
        write(file, "q", qAng)
        #write(file, "tconv", tconv)
        write(file, "conv_signal", conv)
        #write(file, "kernel", K)
        close(file)

    else
        return conv, qAng
    end
end

function wrapper()
    signal = rand(670, 2001)
    tvec = 0:0.5:1000
    fwhm = 150.0
    Flag = 0
    conv = convolve(signal, tvec, fwhm, Flag)
end


function Gaussian(x:: StepRangeLen{Float64}, x0:: Float64, fwhm:: T) where {T<:Union{Float64, Int64}}
    sigma = fwhm/2sqrt(2log(2))
    height = 1/sigma*sqrt(2pi)
    Fx = height .* exp.( (-1 .* (x .- x0).^2)./ (2*sigma^2))
    Fx = Fx./sum(Fx)
end


function Lorentzian(x:: StepRangeLen{Float64}, x0:: Float64, fwhm:: T) where {T<:Union{Float64, Int64}}
    gamma = fwhm/2
    Fx = 1 ./ ((pi * gamma) .* (1 .+ ((x .- x0)./ gamma).^2 ))
    Fx = Fx./sum(Fx)
end


function generate_kernel(tconv:: StepRangeLen{Float64}, fwhm:: Float64, centre:: Float64, ctype:: Int64)
    if ctype == 0
        K = Gaussian(tconv, centre, fwhm)
    elseif cytpe == 1
        K = Lorentzian(tconv, centre, fwhm)
    else
        error("Only Gaussian and Lorentzian functions currently available.")
    end
    K = Matrix(Circulant(K))
    return K
end


function convolve(S:: Matrix{Float64}, time:: StepRangeLen{Float64}, fwhm:: Float64, ctype:: Int64) 
    nr:: Int64, nc:: Int64 = size(S)
    dt:: Float64 = sum(diff(time, dims=1))/(length(time)-1)
    nt:: Int64 = length(time)
    duration:: Float64 = fwhm*3 # must pad three time FHWM to deal with edges
    tmin:: Float64 = minimum(time); tmax:: Float64 = maximum(time)
    if dt != diff(time, dims=1)[1] ; error("Non-linear time range."); end
    if nr != nt && nc != nt; error("Inconsistent dimensions."); end
    if nc != nt; S = copy(transpose(S)); nr, nc = nc, nr; end # column always time axis

    tconv_min:: Float64 = tmin - duration ; tconv_max:: Float64 = tmax + duration
    tconv = tconv_min:dt:tconv_max # extended convoluted time vector
    ntc = length(tconv)
    padend:: Int64 = (duration/dt) # index at which 0 padding ends and signal starts
    padstart:: Int64 = (padend + tmax / dt) # index where signal ends and padding starts

    K = generate_kernel(tconv, fwhm, tconv[padend], ctype)

    sC = zeros(Float64, nr, ntc)
    sC[1:nr, 1:padend] .= @view S[1:nr, 1] # extended and pad signal
    sC[1:nr, padend:padstart] .= S
    sC[1:nr, padstart:end] .= @view S[1:nr, end]

    conv = zeros(Float64, nr, ntc)
    conv = convolution_integral(sC, K, dt)

    S .= conv[:, 1:nt] # return convoluted signal w same original size

    return S 
    end


function convolution_integral(signal:: Matrix{Float64}, kernel:: Matrix{Float64}, dt:: Float64)
    LinearAlgebra.BLAS.set_num_threads(Sys.CPU_THREADS)
    conv = signal * kernel
    conv .*= dt
    return conv 
end

