
using Random
using ArgParse
using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using MLDatasets
using MultivariateStats
using JLD, FileIO

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
            default = 2e-3 
        "--T"
            help = "temperature"
            arg_type = Float64
            default = 1.0
        "--nsamples"
            help = "number of samples"
            arg_type = Int64
            default = 5000
        "--burnin"
            help = "number of samples as burnin"
            arg_type = Int64
            default = 0
        "--q_db"
            help = "probability from dark to bright"
            arg_type = Float64
            default = 1e-1
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
    train_x = train_x.*reshape((2*train_y.-1),(1,size(train_y,1)))
    return (train_x, train_y, test_x, test_y)
end

function run_sampler(args::Dict, X::Array{Float64,2}, y::Array{Int64,1}, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    sampler = RandomWalk(X, y, args["stepsize"], args["pca_dim"], args["T"], args["q_db"])
    samples, total_bs, avg_accept_prob, acc, acc_time, datause = flymc_train(sampler, X, y, args["nsamples"], args["stepsize"], args["pca_dim"], args["T"], args["q_db"], args["burnin"], test_x, test_y)
    return samples, total_bs, avg_accept_prob, acc, acc_time, datause
end

struct RandomWalk
    X::Array{Float64,2}
    y::Array{Int64,1}
    stepsize::Float64
    data_size::Int64
    pca_dim::Int64
    T::Float64
    q_db::Float64
    xi::Float64
    a::Float64
    b::Float64
    c::Float64
    A::Array{Float64,2}
    B::Array{Float64,1}
    theta_prime::Array{Float64,1}
end

function flymc_train(sampler::RandomWalk, X::Array{Float64,2}, y::Array{Int64,1}, nsamples::Int64, stepsize::Float64, pca_dim::Int64, T::Float64, q_db::Float64, burnin::Int64, test_x::Array{Float64,2}, test_y::Array{Int64,1})
    theta = kaiming_unif_init(pca_dim)
    succ = 0.
    samples = zeros(pca_dim, nsamples)
    total_bs = 0.
    iters = nsamples+burnin
    z = Auxiliary(collect(1:size(X,2)), 0)
    interval = 500
    K = Int(iters/interval)
    acc = zeros(K)
    acc_time = zeros(K)
    datause = zeros(K)
    k = 1
    total_runtime = 0.
    for i = 1:iters
        runtime = @elapsed begin
            (theta, z, sig, bs) = next(sampler, theta, z)
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

mutable struct Auxiliary
    # the first num_bright entries of indices are bright; the rest are dark
    indices::Array{Int64,1}
    num_bright::Int64
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

function RandomWalk(X::Array{Float64,2}, y::Array{Int64,1}, stepsize::Float64, pca_dim::Int64, T::Float64, q_db::Float64) 
    EPS = 1e-6
    xi = 1.5
    pexp = exp(xi)
    nexp = exp(-xi)
    f = pexp + nexp
    a = -0.25/xi*(pexp-nexp)/(2 + f)
    b = 0.5
    c = -a*xi^2 - 0.5*log(2 + f) - EPS
    A = X*transpose(X.*a) # (d,d)
    B  = sum(X.*b, dims=2)[:,1] # (d,)
    theta_prime = zeros(pca_dim)
    return RandomWalk(X, y, stepsize, size(X, 2), pca_dim, T, q_db, xi, a, b, c, A, B, theta_prime);
end

function next(self::RandomWalk, theta::Array{Float64,1}, z::Auxiliary)
    data_size = self.data_size
    sig = 0
    theta_prime = proposal(self, theta)
    z = zstep(self, theta ,z)
    bs = z.num_bright
    log_ll_old, log_ll_new = joint_logtarget(self, theta, theta_prime, z)
    logmh = (log_ll_new - log_ll_old)
    acc_prob = stand_mh(self, logmh)
    if rand() < acc_prob
        theta .= theta_prime
        sig = 1
    end
    return (theta, z, sig, bs)
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

function logH(x::Float64)
    abs_x = abs(x)
    return 0.5 * (x - abs_x) - log(1+exp(-abs_x))
end

function joint_logtarget(self::RandomWalk,theta::Array{Float64,1},theta_prime::Array{Float64,1},z::Auxiliary)
    prior = dot(theta, self.A*theta) + dot(self.B, theta) # log_prod_B
    prior_prime = dot(theta_prime, self.A*theta_prime) + dot(self.B, theta_prime)
    log_ll = 0.0
    log_ll_prime = 0.0
    for idx in 1:z.num_bright
        log_ll += log_bd_ratio(self, theta, z.indices[idx])
        log_ll_prime += log_bd_ratio(self, theta_prime, z.indices[idx])
    end
    return (log_ll+prior/self.T, log_ll_prime+prior_prime/self.T)
end

function log_bd_ratio(self::RandomWalk,theta::Array{Float64,1},idx::Int64)
    Xi_dot_theta = 0.0;
    for j = 1:length(theta)
        Xi_dot_theta += self.X[j,idx] * theta[j]
    end
    logl = logH(Xi_dot_theta)
    logBn = self.a*Xi_dot_theta^2 + self.b*Xi_dot_theta + self.c
    gap = (logl - logBn) / self.T
    return gap+log(1-exp(-gap))
end

function zstep(self::RandomWalk,theta::Array{Float64,1},z::Auxiliary)
    initial_num_bright = z.num_bright
    # Consider the bright -> dark proposals
    i = 1
    total = 0
    while (i <= z.num_bright)
        idx = z.indices[i]
        total+=idx
        log_p_accept = log(self.q_db) - log_bd_ratio(self, theta, idx)
        if log(rand()) <  log_p_accept
            (z.indices[i], z.indices[z.num_bright]) = (z.indices[z.num_bright], z.indices[i])
            z.num_bright -= 1
        else
            i += 1
        end
    end
    # Consider the dark -> bright proposals
    i = initial_num_bright+1
    dg = Geometric(self.q_db)
    i += rand(dg)
    while (i <= self.data_size)
        idx = z.indices[i]
        log_p_accept = - log(self.q_db) + log_bd_ratio(self, theta, idx)
        if log(rand()) <  log_p_accept
            z.num_bright += 1
            (z.indices[i], z.indices[z.num_bright]) = (z.indices[z.num_bright], z.indices[i])
        end
        i += 1 + rand(dg)
    end
    return z
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