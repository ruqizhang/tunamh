
using Random
using ArgParse
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using MLDatasets
using MultivariateStats
using JLD, FileIO

include("../util/AliasSampler.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--pca_dim"
            help = "pca dim"
            arg_type = Int64
            default = 50
        "--stepsize"
            help = "stepsize"
            arg_type = Float64
            default = 1e-4
        "--T"
            help = "temperature"
            arg_type = Float64
            default = 1.0
        "--nsamples"
            help = "number of samples"
            arg_type = Int64
            default = 600000
        "--burnin"
            help = "number of samples as burnin"
            arg_type = Int64
            default = 0
    end

    return parse_args(s)
end

function main() 
    args = parse_commandline()
    train_x, train_y, test_x, test_y = generate_data(args)
    samples, total_bs, avg_accept_prob, acc, acc_time, datause = run_sampler(args,train_x,train_y,test_x,test_y)
    println("accuracy: $(acc[end])")
    println("runtime: $(acc_time[end])")
    println("Avg Batch Size: $(total_bs/args["nsamples"])")
    println("Avg Acceptance Prob: $(avg_accept_prob)")
end

function generate_data(args::Dict)
    Random.seed!(222)
    train_x, train_y = MNIST.traindata()
    train_x = reshape(train_x, 784, :)
    idx1 = findall(x -> x == 7, train_y)
    idx2 = findall(x -> x == 9, train_y)
    idx = sort(vcat(idx1, idx2))
    train_y[idx1] .= 0
    train_y[idx2] .= 1
    train_y = train_y[idx]
    train_x = train_x[:,idx]

    test_x, test_y = MNIST.testdata()
    test_x = reshape(test_x, 784, :)
    idx1 = findall(x -> x == 7, test_y)
    idx2 = findall(x -> x == 9, test_y)
    idx = sort(vcat(idx1, idx2))
    test_y[idx1] .= 0
    test_y[idx2] .= 1
    test_y = test_y[idx]
    test_x = test_x[:,idx]
    train_x = convert(Array{Float64},train_x)
    test_x = convert(Array{Float64},test_x)
    M = fit(PCA, train_x; maxoutdim=args["pca_dim"])
    train_x = transform(M, train_x)
    test_x = transform(M, test_x)
    return (train_x, train_y, test_x, test_y)
end

function run_sampler(args::Dict, X::Array{Float64,2}, y::Array{Int64,1}, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    sampler = RandomWalk(X, y, args["stepsize"], args["pca_dim"], args["T"])
    samples, total_bs, avg_accept_prob, acc, acc_time, datause = smh_train(sampler, X, y, args["nsamples"], args["stepsize"], args["pca_dim"], args["T"], args["burnin"], test_x, test_y)
    return samples, total_bs, avg_accept_prob, acc, acc_time, datause
end

struct RandomWalk
    X::Array{Float64,2}
    y::Array{Int64,1}
    stepsize::Float64
    data_size::Int64
    pca_dim::Int64
    c1::Float64
    psi::Array{Float64,1}
    Psi::Float64
    gamma::AbstractWeights
    gamma_A::AliasSampler
    T::Float64
    theta_prime::Array{Float64,1}
end

function smh_train(sampler::RandomWalk, X::Array{Float64,2}, y::Array{Int64,1}, nsamples::Int64, stepsize::Float64, pca_dim::Int64, T::Float64, burnin::Int64, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    theta = kaiming_unif_init(pca_dim)
    succ = 0.
    samples = zeros(pca_dim, nsamples)
    total_bs = 0.
    iters = nsamples+burnin
    interval = 10000
    K = Int(iters/interval)
    acc = zeros(K)
    acc_time = zeros(K)
    datause = zeros(K)
    k = 1
    total_runtime = 0.
    for i = 1:iters
        runtime = @elapsed begin
            (theta, sig, bs) = next(sampler, theta)
            total_bs += bs
            succ += sig
            if i > burnin
                for j = 1:pca_dim
                    samples[j,i-burnin] = theta[j]
                end
            end
        end
        total_runtime += runtime
        if (i % interval == 0)
            acc[k] = test(samples[:,1:i],test_x,test_y)
            acc_time[k] = total_runtime
            datause[k] = total_bs
            k += 1
        end
    end
    avg_accept_prob = float(succ) / iters
    return samples, total_bs, avg_accept_prob, acc, acc_time, datause
end

function kaiming_unif_init(pca_dim::Int64)
    a = sqrt(5.0)
    fan = pca_dim
    gain = sqrt(2.0 / (1 + a^2))
    std = gain / sqrt(fan)
    bound = sqrt(3.0) * std
    theta = 2*bound*rand(pca_dim).-bound
    return theta
end

function RandomWalk(X::Array{Float64,2}, y::Array{Int64,1}, stepsize::Float64, pca_dim::Int64, T::Float64)
    c1 = 1.0
    psi = c1 * sqrt.(vec(sum(X.^2; dims=1)))/T
    Psi = sum(psi)
    gamma = Weights(psi ./ Psi)
    theta_prime = zeros(pca_dim)
    gamma_A = AliasSampler(gamma)
    return RandomWalk(X, y, stepsize, size(X, 2), pca_dim, c1, psi, Psi, gamma, gamma_A, T, theta_prime);
end

function next(self::RandomWalk, theta::Array{Float64,1})
    data_size = self.data_size
    sig = 0
    theta_prime = proposal(self, theta)
    phi = dist(theta_prime, theta)
    lam = phi * self.Psi
    s = rand(Poisson(lam))
    bs = 0
    logmh = 0.0
    idxs = zeros(Int64, s)
    for ii = 1:s
        idx = rand(self.gamma_A)
        (ll_old, ll_new) = logtarget(self, theta, theta_prime, idx)
        lam_i = ll_new - ll_old 
        bs += 1
        if lam_i < 0
            lam_i = -lam_i
            lam_i_bar = phi * self.psi[idx]
            p = lam_i/lam_i_bar
            if rand() < p
                return (theta, 0, bs)
            end
        end
    end
    theta .= theta_prime
    return (theta, 1, bs)
end

function dist(x::Array{Float64,1}, y::Array{Float64,1})
    @assert(length(x) == length(y))
    acc = 0;
    for i = 1:length(x)
        acc += (x[i] - y[i])^2
    end
    return sqrt(acc)
end

function proposal(self::RandomWalk, theta::Array{Float64,1})
    for i = 1:length(theta)
        self.theta_prime[i] = theta[i] + self.stepsize * randn()
    end
    return self.theta_prime
end

function stand_mh(self::RandomWalk, u::Float64)
    return exp(u)
end

function sigmoid(z::Real) 
    return one(z) / (one(z) + exp(-z))
end

function logH(predict::Float64, y::Int64) 
    return y*log(predict) + (1-y)*log(1-predict)
end

function logtarget(self::RandomWalk,theta::Array{Float64,1},theta_prime::Array{Float64,1},idx::Int64)
    Xi_dot_theta = 0.0;
    Xi_dot_theta_prime = 0.0;
    for j = 1:length(theta)
        Xi_dot_theta += self.X[j,idx] * theta[j]
        Xi_dot_theta_prime += self.X[j,idx] * theta_prime[j]
    end
    predict = sigmoid(Xi_dot_theta)
    predict_prime = sigmoid(Xi_dot_theta_prime)
    yi = self.y[idx]
    logl = logH(predict, yi) / self.T
    logl_prime = logH(predict_prime, yi) / self.T
    return (logl, logl_prime)
end

function test(samples::Array{Float64,2}, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    avg_sample = mean(samples, dims=2)
    N = size(test_x, 2)
    acc = 0.0
    for i = 1:N
        predict = dot(avg_sample, test_x[:,i])
        if predict > 0 
            if test_y[i] == 1
                acc += 1.0
            end
        else
            if test_y[i] == 0
                acc += 1.0
            end
        end
    end
    return acc/N
end

main()