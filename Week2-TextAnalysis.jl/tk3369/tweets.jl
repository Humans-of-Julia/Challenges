### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ c8736c18-1f88-11eb-1025-91cf3d825edf
using Pkg

# ╔═╡ 490206e6-1f43-11eb-39db-05eb9f02b2b2
begin
	using TextAnalysis
	using CSV
	using DataFrames
	using Pipe: @pipe
	using Plots
end

# ╔═╡ 4d1d0a34-2478-11eb-3c8e-b1e62b58dcff
using TextAnalysis: NaiveBayesClassifier, fit!, predict

# ╔═╡ c251d572-247e-11eb-29d9-3d382c09f425
begin
	using HTTP
	using JSON3
end

# ╔═╡ 9ecf464a-1ff4-11eb-2c0a-dfa5efc8e305
md"""
# Sentiment analysis using TextAnalysis.jl

This notebook explores the [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) data set from Kaggle.
We will use [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) as the primary tool for analyzing textual data.
"""

# ╔═╡ f500ca32-1f88-11eb-0f34-2d84f8316de7
dir = "/Users/tomkwong/Julia/HumansOfJulia-WeeklyContest/Week2-TextAnalysis.jl/tk3369"

# ╔═╡ cbed3fd6-1f88-11eb-1267-d172e19f441f
Pkg.activate(dir)

# ╔═╡ ff756d94-1ff4-11eb-25ae-5983ea00b7e3
md"""
## Loading data
"""

# ╔═╡ ffc6a542-1f88-11eb-284b-056a0af0a139
cd(dir)

# ╔═╡ f13c9a7a-1f88-11eb-3c7c-5371c8fe8203
df = DataFrame(CSV.File("data/Tweets.csv"))

# ╔═╡ e29c55f0-1ff1-11eb-2f0c-97d6c9fb83c3
md"""
## Data Wrangling

We will take a look at the data and get a little more understanding about what's going on in this data set.
"""

# ╔═╡ ea7dadbc-1ff1-11eb-2397-1fe07e2899be
describe(df, :eltype, :nmissing, :first => first)

# ╔═╡ 2fa37940-1ff3-11eb-01ef-813c561d5679
let x = combine(groupby(df, :airline_sentiment), nrow)
	bar(x.airline_sentiment, x.nrow;
		title = "Airline Sentiment",
		label = :none,
		legend = :topright)
end

# ╔═╡ 4797ae3c-1ff2-11eb-3f31-f32f365ea0ee
histogram(df.airline_sentiment_confidence;
	legend = nothing, 
	title = "Airline Sentiment Confidence")

# ╔═╡ 747a707c-1ff4-11eb-2681-6b6787174d0e
let x = combine(groupby(dropmissing(df, :negativereason), :negativereason), nrow)
	bar(x.negativereason, x.nrow;
		title = "Negative Reasons", label = :none, xrotation = 45)
end

# ╔═╡ 5ca1e714-1f8b-11eb-30fd-dbdd6e574805
md"""
# Examining tweets

The CSV file contains over 14,000 tweets. Let's quickly examine some individual data.
"""

# ╔═╡ 097e9b2c-1ff4-11eb-255d-1d0816f3ddee
md"""
Before we go further, it would be nice to display a single record in table format. We can define a `table` function that converts an indexable object into Markdown format, which can be displayed in this Pluto notebook.
"""

# ╔═╡ 74192f10-1fe0-11eb-3eea-1d9cbc0c4c3b
function table(nt)
	io = IOBuffer()
	println(io, "|name|value|")
	println(io, "|---:|:----|")
	for k in keys(nt)
		println(io, "|`", k, "`|", nt[k], "|")
	end
	return Markdown.parse(String(take!(io)))
end;

# ╔═╡ 32974c34-1ff4-11eb-18ea-67999c24906a
md"""
Here, we will define a variable called `row` and bind it to a slider for quick experimentation.
"""

# ╔═╡ e97bc70a-2002-11eb-3f57-ab1838225b0e
@bind row html"""<input type="range" min="1" max="100" value="36"/>"""

# ╔═╡ db9272f6-1ff3-11eb-12e7-0fa2c8837975
"Current Record: $row"

# ╔═╡ cbcac7e6-1fe0-11eb-0820-9f4c6f3fa51f
table(df[row, :])

# ╔═╡ dd832f9c-1f8a-11eb-001b-3d8be911d80a
md"""
As an example, record #36 has the tweet text as:
```
Nice RT @VirginAmerica: Vibe with the moodlight from takeoff to touchdown. #MoodlitMonday #ScienceBehindTheExperience http://t.co/Y7O0uNxTQP
```

This is a tricky one because it contains all of the followings:
- mention (`@VirginAmeria`)
- hash tag (`#MoodlitMonday` and `#ScienceBehindTheExperience`)
- URL (`http://t.co/Y7O0uNxTQP`)

Technically `RT` is a shorthand for "retweet" so perhaps it should be expanded but let's not worry about that for now.
"""

# ╔═╡ b231f13c-1ff0-11eb-278d-43db738ca57f
md"""
## Handling mentions and hashtags
"""

# ╔═╡ b636fc32-1ff5-11eb-048e-cbc4a77c5f75
md"""
If we just ignore these problems then it can be a disaster.
"""

# ╔═╡ 6cb54136-1f8c-11eb-21d0-1542ddd12437
let s = df[36, :text]
	sd = StringDocument(lowercase(s))
	op = 0x00
	op |= strip_punctuation
	op |= strip_stopwords
	op |= strip_html_tags
	prepare!(sd, op)
	stem!(sd)
	table(ngrams(sd))
end

# ╔═╡ cd6619b0-1ff5-11eb-19f5-092f2f06b172
md"""
Right off the bat, I can see some problems here. It seems that when I stripped punctuations, it also took the `@` and `#` signs away. The URL also became weird.  Oh yeah, that's what stripping punctuation means, right? :-)
"""

# ╔═╡ 8a3e8bbc-1ff6-11eb-2a12-c3fba022267b
md"""
##### Extracting mentions, hash tags, and URL's. 
This neat idea came from José Bayoán Santiago Calderón when I asked the question on Slack. Let's define some functions using regular expressions.
"""

# ╔═╡ c72afbc0-2474-11eb-3f52-e76f60ecd80d
const regexp = Dict(
	:mention => r"@\w+",
	:hashtag => r"#\w+",
	:url => r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
);

# ╔═╡ fd1a4328-1ff7-11eb-0b7f-2d62c0e9d0df
function extract_tokens(s, token_type)
	return collect(x.match for x in eachmatch(regexp[token_type], s))
end

# ╔═╡ b3b62480-1ff8-11eb-3ef9-7985882f19d4
function remove_tokens(s)
	for re in values(regexp)
		s = replace(s, re => "")
	end
	return s
end

# ╔═╡ d4ea52ea-24b0-11eb-11b1-f3079e437485
md"##### Create a new data frame with extracted and clean text fields"

# ╔═╡ 457104ae-1ff8-11eb-04d8-536205dc3fc9
begin
	df2 = DataFrame()
	df2.airline_sentiment = df.airline_sentiment
	df2.text = df.text
	df2.mentions = extract_tokens.(df.text, :mention)
	df2.hashtags = extract_tokens.(df.text, :hashtag)
	df2.urls = extract_tokens.(df.text, :url)
	df2.clean_text = lowercase.(remove_tokens.(df.text))
	df2
end;

# ╔═╡ 91e59d40-1ff8-11eb-0fab-df3732a771c2
table(df2[36, :])

# ╔═╡ de6d7008-215a-11eb-1082-dd3adfb676e6
md"""
As you can see, the mentions/hashtags/urls are extracted into separate columns in the data frame. The `clean_text` field contains the cleaned version of `text`.
"""

# ╔═╡ 5c30f168-2124-11eb-3343-79f2765bd382
md"""
## Using Naive Bayes Classifier
"""

# ╔═╡ 6ebd952a-2124-11eb-30ee-2b80714ccb11
md"""
In our data frame, we already have a column `x_string_doc` with `StringDocuments` values. So we can just fit them to the classifier.
"""

# ╔═╡ 238efc32-1ff9-11eb-3887-17016268dfe2
function create_string_doc(s)
	sd = StringDocument(s)
	op = 0x00
	op |= strip_punctuation
	op |= strip_stopwords
	op |= strip_html_tags
	prepare!(sd, op)
	stem!(sd)
	return sd
end

# ╔═╡ 76ab55ba-2124-11eb-3999-71ce72708dd3
model = let 
	classes = unique(df2.airline_sentiment)
	nbc = NaiveBayesClassifier(classes)
	for (clean_text, class) in zip(df2.clean_text , df2.airline_sentiment)
		sd = create_string_doc(clean_text)
		fit!(nbc, sd, class) 
	end
	nbc
end;

# ╔═╡ 983678c6-2125-11eb-07a7-29321450d287
md"Let's create a model test function and then try our predictor for a few simple test cases."

# ╔═╡ cdcc1764-212c-11eb-1a24-cb5f0aa323a5
function test_model(model, tweets)
	df = DataFrame(text = tweets)
	df.doc = TextAnalysis.text.(create_string_doc.(remove_tokens.(tweets)))
	df.analysis = predict.(Ref(model), df.doc)
	
	df.positive = getindex.(df.analysis, "positive")
	df.negative = getindex.(df.analysis, "negative")
	df.neutral = getindex.(df.analysis, "neutral")
	
	select!(df, Not(:analysis))
	
	return df
end;

# ╔═╡ 058b9f92-2486-11eb-3853-d7f83cf90198
let
	tweets = [
		"whatever airline sucks!", 
		"i love @american service :-)", 
		"just ok", 
		"hello world"]
	test_model(model, tweets)
end

# ╔═╡ 02494270-2130-11eb-1342-390a8240d584
md"""
## Determining accuracy

How well does the Naive Bayes Classifier work?
"""

# ╔═╡ b4aa64ac-2136-11eb-142f-5dfb445e2e96
md"""
As the `predict` function returns a `Dict` object with the probabilities assigned to each class, we need to choose the best option. Let's define a function for that.
"""

# ╔═╡ 54a76b44-2130-11eb-2680-211869c363da
function predict_and_choose(c::NaiveBayesClassifier, sd::StringDocument)
	val = predict(c, sd)
	return argmax(val)
end;

# ╔═╡ 0d66ccd4-2137-11eb-1f8e-2d03c16e29d9
md"Now, make prediction over all 14K tweets."

# ╔═╡ 0fbd02e8-2130-11eb-1f17-d77c0e80fa2a
yhat = let sds = create_string_doc.(lowercase.(remove_tokens.(df2.text)))
	predict_and_choose.(Ref(model), sds)
end;

# ╔═╡ f27be3ec-2130-11eb-0114-33bf20368dba
hits = count(df2.airline_sentiment .== yhat)

# ╔═╡ 55f555f6-2137-11eb-2312-99fbac7dad6e
misses = length(yhat) - hits

# ╔═╡ 669e0fd8-2137-11eb-1eb4-01caefb25d55
wayoff = count(
			(df2.airline_sentiment .!== yhat) .&
			(df2.airline_sentiment .!== "neutral") .&
			(yhat .!== "neutral"))

# ╔═╡ c283de5e-2132-11eb-2cb6-df48c17efd1d
accuracy_percentage = hits / (hits + misses) * 100

# ╔═╡ ed660bee-2132-11eb-3217-a58156b889a9
slightly_off_percentage = (misses - wayoff) / (hits + misses) * 100

# ╔═╡ ffaba636-2132-11eb-1593-63aab54d6c98
way_off_percentage = wayoff / (hits + misses) * 100

# ╔═╡ 398a14bc-2135-11eb-0351-a314bfbb0db4
bar(["hits", "misses slightly", "misses way-off"], 
	[accuracy_percentage, slightly_off_percentage , way_off_percentage];
	legend = :none)

# ╔═╡ b9da38a8-247e-11eb-1fcd-0b19be7ece16
md"# Analyze some random tweets"

# ╔═╡ c57bb7b8-247e-11eb-3055-c7c83a9bfe93
token = readline("/Users/tomkwong/.twitter-bearer");

# ╔═╡ 0863fb44-247f-11eb-3fc9-935c87fdff28
response = HTTP.get("https://api.twitter.com/1.1/search/tweets.json?q=lang%3Aen%20flight", ["authorization" => "Bearer $token"]);

# ╔═╡ 19c808ee-247f-11eb-2082-59aa94a3e77c
data = JSON3.read(response.body);

# ╔═╡ 5a873d72-2485-11eb-10e1-bb119c1e2795
data[:statuses][1]

# ╔═╡ 32fb7e76-2485-11eb-3f0b-6f779c314a81
tweets = [x.text for x in data.statuses]

# ╔═╡ de5600b6-2485-11eb-1111-a729caa3a913
result = test_model(model, tweets)

# ╔═╡ 82d9d400-24a4-11eb-0c28-950af58adee4
table(result[9,:])

# ╔═╡ Cell order:
# ╟─9ecf464a-1ff4-11eb-2c0a-dfa5efc8e305
# ╠═c8736c18-1f88-11eb-1025-91cf3d825edf
# ╠═f500ca32-1f88-11eb-0f34-2d84f8316de7
# ╠═cbed3fd6-1f88-11eb-1267-d172e19f441f
# ╠═490206e6-1f43-11eb-39db-05eb9f02b2b2
# ╟─ff756d94-1ff4-11eb-25ae-5983ea00b7e3
# ╠═ffc6a542-1f88-11eb-284b-056a0af0a139
# ╠═f13c9a7a-1f88-11eb-3c7c-5371c8fe8203
# ╟─e29c55f0-1ff1-11eb-2f0c-97d6c9fb83c3
# ╠═ea7dadbc-1ff1-11eb-2397-1fe07e2899be
# ╠═2fa37940-1ff3-11eb-01ef-813c561d5679
# ╠═4797ae3c-1ff2-11eb-3f31-f32f365ea0ee
# ╠═747a707c-1ff4-11eb-2681-6b6787174d0e
# ╟─5ca1e714-1f8b-11eb-30fd-dbdd6e574805
# ╟─097e9b2c-1ff4-11eb-255d-1d0816f3ddee
# ╠═74192f10-1fe0-11eb-3eea-1d9cbc0c4c3b
# ╟─32974c34-1ff4-11eb-18ea-67999c24906a
# ╠═e97bc70a-2002-11eb-3f57-ab1838225b0e
# ╠═db9272f6-1ff3-11eb-12e7-0fa2c8837975
# ╠═cbcac7e6-1fe0-11eb-0820-9f4c6f3fa51f
# ╟─dd832f9c-1f8a-11eb-001b-3d8be911d80a
# ╟─b231f13c-1ff0-11eb-278d-43db738ca57f
# ╟─b636fc32-1ff5-11eb-048e-cbc4a77c5f75
# ╠═6cb54136-1f8c-11eb-21d0-1542ddd12437
# ╟─cd6619b0-1ff5-11eb-19f5-092f2f06b172
# ╟─8a3e8bbc-1ff6-11eb-2a12-c3fba022267b
# ╠═c72afbc0-2474-11eb-3f52-e76f60ecd80d
# ╠═fd1a4328-1ff7-11eb-0b7f-2d62c0e9d0df
# ╠═b3b62480-1ff8-11eb-3ef9-7985882f19d4
# ╟─d4ea52ea-24b0-11eb-11b1-f3079e437485
# ╠═457104ae-1ff8-11eb-04d8-536205dc3fc9
# ╠═91e59d40-1ff8-11eb-0fab-df3732a771c2
# ╟─de6d7008-215a-11eb-1082-dd3adfb676e6
# ╟─5c30f168-2124-11eb-3343-79f2765bd382
# ╟─6ebd952a-2124-11eb-30ee-2b80714ccb11
# ╠═4d1d0a34-2478-11eb-3c8e-b1e62b58dcff
# ╠═238efc32-1ff9-11eb-3887-17016268dfe2
# ╠═76ab55ba-2124-11eb-3999-71ce72708dd3
# ╟─983678c6-2125-11eb-07a7-29321450d287
# ╠═cdcc1764-212c-11eb-1a24-cb5f0aa323a5
# ╠═058b9f92-2486-11eb-3853-d7f83cf90198
# ╟─02494270-2130-11eb-1342-390a8240d584
# ╟─b4aa64ac-2136-11eb-142f-5dfb445e2e96
# ╠═54a76b44-2130-11eb-2680-211869c363da
# ╟─0d66ccd4-2137-11eb-1f8e-2d03c16e29d9
# ╠═0fbd02e8-2130-11eb-1f17-d77c0e80fa2a
# ╠═f27be3ec-2130-11eb-0114-33bf20368dba
# ╠═55f555f6-2137-11eb-2312-99fbac7dad6e
# ╠═669e0fd8-2137-11eb-1eb4-01caefb25d55
# ╠═c283de5e-2132-11eb-2cb6-df48c17efd1d
# ╠═ed660bee-2132-11eb-3217-a58156b889a9
# ╠═ffaba636-2132-11eb-1593-63aab54d6c98
# ╠═398a14bc-2135-11eb-0351-a314bfbb0db4
# ╟─b9da38a8-247e-11eb-1fcd-0b19be7ece16
# ╠═c251d572-247e-11eb-29d9-3d382c09f425
# ╠═c57bb7b8-247e-11eb-3055-c7c83a9bfe93
# ╠═0863fb44-247f-11eb-3fc9-935c87fdff28
# ╠═19c808ee-247f-11eb-2082-59aa94a3e77c
# ╠═5a873d72-2485-11eb-10e1-bb119c1e2795
# ╠═32fb7e76-2485-11eb-3f0b-6f779c314a81
# ╠═de5600b6-2485-11eb-1111-a729caa3a913
# ╠═82d9d400-24a4-11eb-0c28-950af58adee4
