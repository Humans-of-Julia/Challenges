#=
Motivation: I want to model psychometric data to some kind of sigmoid function, but the nature
of psychometric data is that it is forced choice.  This means that the output is binary.
GLM doesn't support forced choice models yet——however, with a little bit of work and some
type piracy, the results are attractive.
=#

using GLM, DataFrames, CSV, FreqTables, StatsPlots, Distributions, Statistics
const Dist = Distributions
# imports for overloading
import GLM: Link, Link01, linkfun, linkinv, mueta, inverselink

# obtain dataset
df_raw = DataFrame(CSV.File(download("https://gist.githubusercontent.com/jakewilliami/7a84208f33de8076c7b17abb52a9b242/raw/b27d63700f4072830251eb4063e790c27c3154d0/mcd.csv")))

# statistical notations
Φ(x::Real)   = Dist.cdf(Dist.Normal(), x)
Φ⁻¹(x::Real) = Dist.quantile(Dist.Normal(), x)
φ(x::Real)   = Dist.pdf(Dist.Normal(), x)

# define a new link function for 2-alternative forced choice
struct Probit2AFCLink <: Link end

# overloading
linkfun(::Probit2AFCLink, μ::Real) = Φ⁻¹(2 * max(μ, nextfloat(0.5)) - 1)
linkinv(::Probit2AFCLink, η::Real) = (1 + Φ(η)) / 2
mueta(::Probit2AFCLink,   η::Real) = φ(η) / 2
function inverselink(::Probit2AFCLink, η::Real)
    μ = (1 + Φ(η)) / 2
    d = φ(η) / 2
    return μ, d, oftype(μ, NaN)
end

# define models
logit(p) = log(p / (1 - p))
logit⁻¹(α) = 1 / (1 + exp(-α))
logit⁻¹(α) = logit⁻¹(α) * 0.5 + 0.5 # shift and squish for 2-AFC
logit2afc⁻¹(α) = 0.5 + 0.5 / (1 + exp(-α))
probit(p) = Dist.quantile(Dist.Normal(), p)
probit⁻¹(x) = Dist.cdf(Dist.Normal(), x)
Φ⁻¹(z) = probit(z) # statistical notation
Φ(α) = probit⁻¹(α) # statistical notation
probit2afc⁻¹(x) = probit⁻¹(x) * 0.5 + 0.5 # shift and squish for 2-AFC

df_pivot = combine(groupby(df_raw, :condition1)) do df
    μ = mean(df.correct)
    n = nrow(df)
    (
    correct_mean = μ,
    n = n,
    k = μ * n
    )
end

model = glm(@formula(correct ~ condition1) , df_raw, Binomial(), Probit2AFCLink())
a, b = coef(model)

theme(:solarized)
plot = @df df_pivot scatter(
        :condition1,
        :correct_mean,
        title = "Psychometric Curve of Motion Coherence",
        label = false,
        xaxis = "Coherence",
        yaxis = "Accuracy",
        fontfamily = font("Times")
    )

# add models to graph
plot!(plot, x -> probit2afc⁻¹(a + b*x), 0, 0.32, label = "Probit Link")
plot!(plot, x -> logit2afc⁻¹(a + b*x), 0, 0.32, label = "Logit Link")
