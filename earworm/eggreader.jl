using HDF5
using Plots
using FFTW

struct LocustP3Egg
    eggattr
    eggdata
end

function GetLocustP3EggAttr(h5file, path2egg)

    channels = keys(h5file["channels"])
    streams = keys(h5file["streams"])

    eggattr = Dict("nch" => size(channels, 1), "nstrm" => size(streams, 1), "acq" => Dict())

    eggname = split(path2egg, "/")[end]
    for paramstr in ["angle", "radius", "energy", "axial"]
        eggattr[paramstr] = parse(Float64, (split(split(path2egg, paramstr)[2], "_")[1]))
    end

    for stream in streams
        eggattr["acq"][stream] = keys(h5file["streams"][stream]["acquisitions"])
        # println(keys(infile["streams"][stream]["acquisitions"]))
    end

    return eggattr
end

function ReadLocustP3Egg(path2egg, vrange = 5.5e-8, nbit = 8, samplerate = 200e6)
    infile = h5open(path2egg, "r")

    eggattr = GetLocustP3EggAttr(infile, path2egg)

    eggdata = Dict()
    for streamacqpair in eggattr["acq"]
        eggdata[streamacqpair.first] = Dict()
        for acq in eggattr["acq"][streamacqpair.first]

            idata = -vrange / 2 .+ (vrange / 2 ^ nbit) * infile["streams"][streamacqpair.first]["acquisitions"][acq][1:2:end, 1]
            qdata = -vrange / 2 .+ (vrange / 2 ^ nbit) * infile["streams"][streamacqpair.first]["acquisitions"][acq][2:2:end, 1]
            data = transpose(
                            reshape(
                                    idata + 1im * qdata, (size(idata, 1) ÷ eggattr["nch"], eggattr["nch"])
                                    )
                                )
            eggdata[streamacqpair.first][acq] =  data

            eggdata["t"] = collect(0:1:(size(eggdata[streamacqpair.first][acq][1, :], 1) - 1)) * 1 / samplerate
        end
    end
    close(infile)
    return LocustP3Egg(eggattr, eggdata)
end

function WriteP3H5(path2write, vrange = 5.5e-8, nbit = 8, samplerate = 200e6)
    infile = h5open(path2egg, "r")

    eggattr = GetLocustP3EggAttr(infile, path2egg)

    eggdata = Dict()
    for streamacqpair in eggattr["acq"]
        eggdata[streamacqpair.first] = Dict()
        for acq in eggattr["acq"][streamacqpair.first]

            idata = -vrange / 2 .+ (vrange / 2 ^ nbit) * infile["streams"][streamacqpair.first]["acquisitions"][acq][1:2:end, 1]
            qdata = -vrange / 2 .+ (vrange / 2 ^ nbit) * infile["streams"][streamacqpair.first]["acquisitions"][acq][2:2:end, 1]
            data = transpose(
                            reshape(
                                    idata + 1im * qdata, (size(idata, 1) ÷ eggattr["nch"], eggattr["nch"])
                                    )
                                )
            eggdata[streamacqpair.first][acq] =  data

            eggdata["t"] = collect(0:1:(size(eggdata[streamacqpair.first][acq][1, :], 1) - 1)) * 1 / samplerate
        end
    end
    close(infile)
    return LocustP3Egg(eggattr, eggdata)
end



path2egg = "/home/zieglerad/hdd/repos/earworm/earworm/angle86.0000_radius0.000_energy18500.00_axial0.000_locust.egg"

eggfile = ReadLocustP3Egg(path2egg)

gr()
plot()
display(plot!(abs2.(fft(eggfile.eggdata["stream0"]["0"][1,1500:9692]))))
#plot!(imag(eggfile.eggdata["stream0"]["0"])[1,:])
#display(plot!(xlims = (1500,8192 + 1500)))
readline()
#plot(imag(eggfile.eggdata["stream0"]["0"])[1,:])



#println(eggfile)
