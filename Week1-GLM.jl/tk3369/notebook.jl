### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 81915028-18e7-11eb-0aed-57a75839d4a7
begin
	using CSV
	using DataFrames
	using Plots
	using StatsPlots
	using Pipe: @pipe
	using GLM
	using StatsBase
	using Statistics
	using Dates
end

# ╔═╡ 02eb9d36-18e8-11eb-24a9-db6f58080189
md"""
# Seoul Bike Sharing Demand

This notebook demonstrates how to use GLM to analyze the Seoul Bike Sharing Demand
data set.

## Data definitions

The following definitions are captured from the [UCI Datasets Archive page](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand).

- Date: year-month-day
- Rented Bike count: Count of bikes rented at each hour
- Hour: Hour of he day
- Temperature: Temperature in Celsius
- Humidity: %
- Windspeed: m/s
- Visibility: 10m
- Dew point temperature: Celsius
- Solar radiation: MJ/m2
- Rainfall: mm
- Snowfall: cm
- Seasons: Winter, Spring, Summer, Autumn
- Holiday: Holiday/No holiday
- Functional Day: NoFunc(Non Functional Hours), Fun(Functional hours)

As the dataset is targeted for bike sharing demand, the key metrics should be the Rented Bike Count field. The rest of the columns are variables that may be correlated to bike sharing demand.

Taking a quick peek at the variables, it makes sense that they may directly or indirectly affect people's decision of renting a bike. For example, when it's very hot or very cold, then I may take on some other transportation. Likewise, humidity, solar radiation, visibility, rainfall, and snowfall probably play a part as well.

Most variables are continuous variables. The last three columns seasons, holiday, and functional day are discrete variables. As for the hour of day variable, we may want to consider that as discrete because increasing the value (later in the day) does not always affect the response variable (rented bike count) in the same direction.
"""

# ╔═╡ eedb4bf8-19a9-11eb-106b-69b047d4118e
md"""
## Reading data
"""

# ╔═╡ dfa5fbf0-18e7-11eb-391b-1d878b9c96f0
file = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"

# ╔═╡ aed964bc-18e7-11eb-11f1-a3d9957a45f5
df = DataFrame(CSV.File(download(file); dateformats = Dict(:Date => "dd/mm/yyyy")));

# ╔═╡ 4d4a9158-18ec-11eb-2fb0-c178637f903a
md"""
## Data wrangling
"""

# ╔═╡ 09577b1e-19aa-11eb-3e6c-175705b4b951
md"""
Let's quickly examine the data frame and its columns.
"""

# ╔═╡ c5b4e26a-18e7-11eb-2bef-2d937faa252d
describe(df, :eltype, :min, :mean, :max, :nmissing)

# ╔═╡ 6ddeb944-18ec-11eb-05c7-b51e7b01578e
md"It looks pretty clean in general but let's rename some columns so they can be referenced more easily in the code below."

# ╔═╡ ebc9c09a-18e9-11eb-1f56-ef92258af11f
rename!(df, 
	2 => :RentedBikeCount,
	4 => :Temperature, 5 => :Humidity, 6 => :WindSpeed, 7 => :Visibility,
	8 => :DewPointTemperature, 9 => :SolarRadiation, 10 => :Rainfall, 11 => :Snowfall,
	14 => :FunctioningDay
);

# ╔═╡ 16c31048-19b6-11eb-2ad8-1f984acfb06f
md"""
## Correlation analysis
"""

# ╔═╡ 9564ab04-19b4-11eb-3c4c-815fe8a6031a
names(df)

# ╔═╡ 3084d7a6-19b5-11eb-1fa5-9b9f7cd482f4
let 
	cols = 3:11
	colnames = names(df)[cols]
	Σ = cor(Matrix(df[:, cols]))
	heatmap(colnames, colnames, Σ; xrotation = 45, seriescolor = :PiYG_4)
end

# ╔═╡ 2ef9d8a6-19b6-11eb-0cc5-e1831ce6c40a
md"""
It appears that temperature and dew point temperature columns are highly correlated. For that reason, we may want to exclude dew point temperature in our model below.
"""

# ╔═╡ 7deee71e-18ec-11eb-113d-fb0cdc6e07ef
md"""
## Seasonal analysis
Since this is a time series, let's check if there's a trend.
"""

# ╔═╡ 32cd8006-18ec-11eb-221a-d38281aadffb
plot(df.Date, df.RentedBikeCount; 
	legend = :none, 
	ylabel = "Rented Bike Count")

# ╔═╡ a9f3aeda-18ec-11eb-2cb9-9f12fd001d88
md"""
As we only have 1 year of data, we cannot really see any seasonal trends. However, my intuition is that people don't tend to ride bikes during winter due to the low temperature. When spring comes around March/April, the demand picked up.

Now, December 2018 still look like a higher demand when compared with December 2017. A possibility is that biking may have gotten popular? Not sure...
"""

# ╔═╡ 69199458-18ef-11eb-1a46-7d71379a4c04
md"""
## Time of day

My initial guess is that people probably do not rent bikes early morning or late night. Let's verify that.
"""

# ╔═╡ 5de23772-18ed-11eb-1105-85661397bcd3
let 
	data = @pipe df |> 
		select(_, :Date, :Hour, :RentedBikeCount) |>
		unstack(_, :Hour, :RentedBikeCount) |>
		sort(_, :Date)
	
	matrix = Matrix(data[:, 2:end])'
	
	heatmap(data[:, 1], 0:23, matrix;
		title = "Rented Bike Count by Date / Hour",
		ylabel = "Hour",
		xrotation = 45.0,
		xticks = Date(2017,12,1):Month(1):Date(2019,1,1),
		yticks = 0:4:24)
end

# ╔═╡ 8789d48e-18ef-11eb-2b87-2337472759c7
md"""
**The two horizontal stripes show that 8 AM and 6 PM are popular hours for bike rentals.** That makes sense because people might want to do some exercise right before work or after work. Or, they may use it for transportation.

**Korean seems to stay up late.** Bike rental demand continues after 6 PM until almost midnight. By 3 AM, it's total silence.

Now, let's take a look at the same data from a different angle using a boxplot.
"""

# ╔═╡ 5ef37b5a-18eb-11eb-3f75-cfb5947950fb
begin
	hline([1800], color = :green)
	@df df StatsPlots.boxplot!(:Hour, :RentedBikeCount; 
		legend = :none, 
		color = palette(:default)[1],  # reset to first color due to `hline` above
		xlabel = "Hour", 
		ylabel = "Rented Bike Count")
end

# ╔═╡ 3dc1127e-19a4-11eb-2c09-4f2624f479ac
md"""
The boxplot shows the quartiles for each hour across all dates in the year. It confirms our findings that 8 AM and 6 AM are popular times.
"""

# ╔═╡ 716cb212-19b1-11eb-190a-7702595f6bab
md"""
## Categorical Variables

Some variables should be converted to categorical such that GLM.jl can encode dummy variables as such. Instead of mutating the existing `Hour` column, I will create a new `Hour2` column and make it categorical.
"""

# ╔═╡ 79056460-19b6-11eb-1c5c-bf9224a0b18e
df.Hour2 = df.Hour

# ╔═╡ 9506548a-19b6-11eb-02dc-ad2832999a5b
categorical!(df, :Hour2);

# ╔═╡ 7553988e-19a5-11eb-3549-bfd4ea98b2fd
md"""
## Linear Regression

Using linear regression, we can fit the data to a linear equation. Here, the response variable is `RentedBikeCount`. We can choose any of the other fields as explanatory variables. Let's start with something simple - an ordinary linear model.
"""

# ╔═╡ c18fa04e-19a5-11eb-2f5c-0bdba322a26f
hour_model = @formula(RentedBikeCount ~ 1 + Hour2)

# ╔═╡ 558893b4-19a6-11eb-3e84-ef27b2749b80
md"""
To fit the model, we can use the `lm` function.
"""

# ╔═╡ 66b6f680-19a6-11eb-2d2f-afe8d206ee5f
ols = lm(hour_model, df)

# ╔═╡ a46a544a-19a6-11eb-3de3-87dcf7bf431f
md"""
We can now use the fitted model to predict bike demand.
"""

# ╔═╡ bfe39e20-19a6-11eb-1b9c-b95be62dadb5
let 
	y = df.RentedBikeCount
	yhat = round.(Int, predict(ols))
	(r_squared = r2(ols), rmsd = rmsd(y, yhat))
end

# ╔═╡ 079fe824-19a7-11eb-3f1d-b3aa9c9b0f33
md"""
The $R^2$ is very low, meaning that the current model cannot make very accurate prediction. That's understandable because we have only used a single explanatory variable. 

The root mean squared deviation (`rmsd`) value shows how much the predicted values deviates from the actual values.

Now, let's design a more complex model but we will continue to use an ordinary linear model.
"""

# ╔═╡ 41fb7ff4-19a7-11eb-0945-017db54aa411
let 
	model = @formula(RentedBikeCount ~
						1 + Hour2 + Temperature + Humidity + WindSpeed +
						Visibility + SolarRadiation + Rainfall + Snowfall + 
						Seasons + Holiday + FunctioningDay)
	fitted = lm(model, df)
	y = df.RentedBikeCount
	yhat = round.(Int, predict(fitted))
	(r_squared = r2(fitted), rmsd = rmsd(y, yhat))
end

# ╔═╡ e3615da0-19a7-11eb-3dc7-d734a6da4420
md"""
That's great result. The $R^2$ has jumped to 0.66 now! The RMSD is also reduced quite significantly from 543 to 375.
"""

# ╔═╡ ab20e2f6-19a9-11eb-09cd-eff9e75608ee
md"""
## Generalized Linear Model (GLM)

I wonder if we can do better. The GLM.jl package supports Generalized Linear Model (GLM) which is more flexible that linear regression. We will demonstrate how it works below.

Given that the response variable is a count, the general wisdom (not mine) is to design the model with Poisson distribution.
"""

# ╔═╡ 1accea7a-19a8-11eb-2e37-b943977ad801
let 
	model = @formula(RentedBikeCount ~
						1 + Hour2 + Temperature + Humidity + WindSpeed +
						Visibility + SolarRadiation + Rainfall + Snowfall + 
						Seasons + Holiday + FunctioningDay)
	fitted = glm(model, df, Poisson(), LogLink())
	y = df.RentedBikeCount
	yhat = round.(Int, predict(fitted))
	(rmsd = rmsd(y, yhat), )
end

# ╔═╡ 61eb2598-19a8-11eb-055d-b166da58fa50
md"""
There is no `r2` function defined for GLM models, so we just show RMSD here. As you can see, RMSD is further reduced using this model. That's an improvement.

What if we build an even more complex model? So far, all variables are independent. If we introduce interaction terms (multiple variables interacting with each other) then we may create a more powerful predictor.

The question is how to choose the right variables for the interaction terms. My gut feeling is that it would be appropriate to choose variables that are "orthogonal" to each other. For example, humidity and rainfall should be highly correlated and the interaction between them would be somewhat uninteresting. Hence, I have chosen to mix hour of day, temperature, and humidity in the following experiment.
"""

# ╔═╡ 66967c54-19b3-11eb-1b13-856ef40132fa
let 
	model = @formula(RentedBikeCount ~
						1 + Hour2 + Temperature + Humidity + WindSpeed +
						Visibility + SolarRadiation + Rainfall + Snowfall + 
						Seasons + Holiday + FunctioningDay +
						Hour2 * Temperature * Humidity
	)
	fitted = glm(model, df, Poisson(), LogLink())
	y = df.RentedBikeCount
	yhat = round.(Int, predict(fitted))
	(rmsd = rmsd(y, yhat), )
end

# ╔═╡ 1de7c9b2-19b4-11eb-2085-7b7383899819
md"""
That's great! The RMSD is now further reduced although not by a whole lot.
"""

# ╔═╡ 34be34aa-19b4-11eb-1a17-e1253c48b050
md"""
## Todo's

So far, I have been fitting the model with the complete data set. In order to test the predictive power of the model, I should test it against unseen data. Of course, I don't have any more data than what I have downloaded. What I should do is to split the data set and do cross validation.
"""

# ╔═╡ 544e7ac0-19b2-11eb-3931-8f7f10c23212
md"""
## Resources

I don't know much about GLM before working on this. I found the following resources useful as I learn about the subject:

- Foundations of Linear and Generalized Linear Models by Alan Agresti (Wiley 2015)
- [Introduction to Generalized Linear Models](https://online.stat.psu.edu/stat504/node/216/)
- [MIT 18.650 Statistics for Applications, Phillipe Rigolle, Lecture 21-22](https://www.youtube.com/watch?v=X-ix97pw0xY)
"""

# ╔═╡ 7c1251d4-19a9-11eb-0450-8dcae5fdc135
md"""
Thanks you for reading. I hope you enjoy this notebook.

_Tom Kwong_, 
_October 2020_
"""

# ╔═╡ Cell order:
# ╟─02eb9d36-18e8-11eb-24a9-db6f58080189
# ╟─eedb4bf8-19a9-11eb-106b-69b047d4118e
# ╠═81915028-18e7-11eb-0aed-57a75839d4a7
# ╠═dfa5fbf0-18e7-11eb-391b-1d878b9c96f0
# ╠═aed964bc-18e7-11eb-11f1-a3d9957a45f5
# ╟─4d4a9158-18ec-11eb-2fb0-c178637f903a
# ╟─09577b1e-19aa-11eb-3e6c-175705b4b951
# ╠═c5b4e26a-18e7-11eb-2bef-2d937faa252d
# ╟─6ddeb944-18ec-11eb-05c7-b51e7b01578e
# ╠═ebc9c09a-18e9-11eb-1f56-ef92258af11f
# ╟─16c31048-19b6-11eb-2ad8-1f984acfb06f
# ╠═9564ab04-19b4-11eb-3c4c-815fe8a6031a
# ╠═3084d7a6-19b5-11eb-1fa5-9b9f7cd482f4
# ╟─2ef9d8a6-19b6-11eb-0cc5-e1831ce6c40a
# ╟─7deee71e-18ec-11eb-113d-fb0cdc6e07ef
# ╠═32cd8006-18ec-11eb-221a-d38281aadffb
# ╟─a9f3aeda-18ec-11eb-2cb9-9f12fd001d88
# ╟─69199458-18ef-11eb-1a46-7d71379a4c04
# ╠═5de23772-18ed-11eb-1105-85661397bcd3
# ╟─8789d48e-18ef-11eb-2b87-2337472759c7
# ╠═5ef37b5a-18eb-11eb-3f75-cfb5947950fb
# ╟─3dc1127e-19a4-11eb-2c09-4f2624f479ac
# ╟─716cb212-19b1-11eb-190a-7702595f6bab
# ╠═79056460-19b6-11eb-1c5c-bf9224a0b18e
# ╠═9506548a-19b6-11eb-02dc-ad2832999a5b
# ╟─7553988e-19a5-11eb-3549-bfd4ea98b2fd
# ╠═c18fa04e-19a5-11eb-2f5c-0bdba322a26f
# ╟─558893b4-19a6-11eb-3e84-ef27b2749b80
# ╠═66b6f680-19a6-11eb-2d2f-afe8d206ee5f
# ╟─a46a544a-19a6-11eb-3de3-87dcf7bf431f
# ╠═bfe39e20-19a6-11eb-1b9c-b95be62dadb5
# ╟─079fe824-19a7-11eb-3f1d-b3aa9c9b0f33
# ╠═41fb7ff4-19a7-11eb-0945-017db54aa411
# ╟─e3615da0-19a7-11eb-3dc7-d734a6da4420
# ╟─ab20e2f6-19a9-11eb-09cd-eff9e75608ee
# ╠═1accea7a-19a8-11eb-2e37-b943977ad801
# ╟─61eb2598-19a8-11eb-055d-b166da58fa50
# ╠═66967c54-19b3-11eb-1b13-856ef40132fa
# ╟─1de7c9b2-19b4-11eb-2085-7b7383899819
# ╟─34be34aa-19b4-11eb-1a17-e1253c48b050
# ╟─544e7ac0-19b2-11eb-3931-8f7f10c23212
# ╟─7c1251d4-19a9-11eb-0450-8dcae5fdc135
