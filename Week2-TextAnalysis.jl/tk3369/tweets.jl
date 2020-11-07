### A Pluto.jl notebook ###
# v0.12.7

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
end

# ╔═╡ 3fab77c4-1ff2-11eb-02bf-ed577baacef7
using Plots, StatsPlots

# ╔═╡ 3df9b88c-1fe0-11eb-1032-31d3baa1a91b
using TextAnalysis: strip_punctuation, strip_stopwords, strip_punctuation

# ╔═╡ 2b71ba0e-2123-11eb-0f43-bb72fb34208a
using TextAnalysis: NaiveBayesClassifier, fit!, predict

# ╔═╡ 9ecf464a-1ff4-11eb-2c0a-dfa5efc8e305
md"""
# Sentiment analysis using TextAnalyis.jl

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

# ╔═╡ 4797ae3c-1ff2-11eb-3f31-f32f365ea0ee
histogram(df.airline_sentiment_confidence, legend = nothing, title = "Airline Sentiment Confidence")

# ╔═╡ bec8eb92-1ff2-11eb-033d-ff5d3be4df0a
let x = combine(groupby(df, :airline), nrow)
	bar(x.airline, x.nrow, label = "Airline")
end

# ╔═╡ 2fa37940-1ff3-11eb-01ef-813c561d5679
let x = combine(groupby(df, :airline_sentiment), nrow)
	bar(x.airline_sentiment, x.nrow;
		label = "Airline Sentiment",
		legend = :topleft)
end

# ╔═╡ 747a707c-1ff4-11eb-2681-6b6787174d0e
let x = combine(groupby(dropmissing(df, :negativereason), :negativereason), nrow)
	bar(x.negativereason, x.nrow, label = "Negative Reason", rotation = 45)
end

# ╔═╡ 5248a178-1ff3-11eb-383e-993d9e6287d9
unique(df.negativereason)

# ╔═╡ 5ca1e714-1f8b-11eb-30fd-dbdd6e574805
md"""
# Examining tweets

The Tweets.csv file contains over 14,000 tweets. Let's quickly examine some individual data.
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
	Markdown.parse(String(take!(io)))
end

# ╔═╡ 32974c34-1ff4-11eb-18ea-67999c24906a
md"""
Here, we will define a variable called `row` and bind it to a slider for quick experimentation.
"""

# ╔═╡ e97bc70a-2002-11eb-3f57-ab1838225b0e
@bind row html"""<input type="range" min="1" max="$(nrow(df))" value="36"/>"""

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

This is a tricky one because it contins all of the followings:
- user mention
- hash tag
- URL
- Shorthand such as RT (for retweet)
"""

# ╔═╡ b231f13c-1ff0-11eb-278d-43db738ca57f
md"""
## Handling mentions and hashtags

Because mentions(`@`) and hashtags(`#`) look like punctuations, the standard tokenizer would have taken them away. It would be wrong to consider someone like `@happy` as the same as `["@", "happy"]` because the tweet could have been classified with positive sentiment rather than a neutral stance as it is really just a reference to a user name.

There are several ways to handle this:

1. Use `WordTokenizers.tweet_tokenize` function to tokenize the tweet. This tokenizer is aware of mentions and hashtags and it would keep them intact with the text that follows.

2. Extract mentions and hashtags from the tweet and replace them with empty string.
"""

# ╔═╡ 89533ac8-1ff5-11eb-0838-21d9d514ab8a
md"""
For now, let's go with approach #1. Taking advice from Oxinabox from Slack, I can set the default tokenizer with the tweet tokenizer provided by `WordsTokenizers` package.
"""

# ╔═╡ 04c821e0-1fe0-11eb-3a1b-670c5ff35a90
const WT = TextAnalysis.WordTokenizers;

# ╔═╡ 1c0b8dea-1fe0-11eb-19a2-49a1af2ed48c
WT.set_tokenizer(WT.tweet_tokenize);

# ╔═╡ b636fc32-1ff5-11eb-048e-cbc4a77c5f75
md"""
Let me experiment some preprocessing facilities.
"""

# ╔═╡ 6cb54136-1f8c-11eb-21d0-1542ddd12437
let s = StringDocument(lowercase(df[36, :text]))
	op = 0x00
	op |= strip_punctuation
	op |= strip_stopwords
	op |= strip_html_tags
	prepare!(s, op)
	stem!(s)
	ngrams(s) |> table
end

# ╔═╡ cd6619b0-1ff5-11eb-19f5-092f2f06b172
md"""
Right off the bat, I can see some problems here. It seems that when I stripped punctuations, it also took the mention and hashtag away. Also, the URL became weird. So let me get rid of the `strip_punctuation` preparation step for now.
"""

# ╔═╡ 3247ecdc-1ff6-11eb-2a79-85b10f64a651
let s = StringDocument(lowercase(df[36, :text]))
	op = 0x00
	op |= strip_stopwords
	op |= strip_html_tags
	prepare!(s, op)
	stem!(s)
	ngrams(s) |> table
end

# ╔═╡ 39546492-1ff6-11eb-1848-37e0d8420226
md"""
Now, the mention and hashtag are back to normal. The URL has gotten better but something is missing. It used to be `http://t.co/Y7O0UNXTQP`. So it's missing the `t` in domain name. Apparently, the `strip_stopwords` preparation step took that away.
"""

# ╔═╡ 8a3e8bbc-1ff6-11eb-2a12-c3fba022267b
md"""
I'm torn. How can I make it not mess around with my mentions, hashtags, and URLs? Maybe I will take approach #2 now. I can certainly extract these data first before tokenizing the text.

This idea came from José Bayoán Santiago Calderón from Slack. 
"""

# ╔═╡ 9e3133ea-1ff6-11eb-3857-4f2dd7436ccb
let s = df[36, :text]
	
	mention_regex = r"@\w+"
	hashtag_regex = r"#\w+"
	url_regex = r"http[\w:/.]+"
	
	mentions = collect(x.match for x in eachmatch(mention_regex, s))
	hashtags = collect(x.match for x in eachmatch(hashtag_regex, s))
	urls = collect(x.match for x in eachmatch(url_regex, s))
	
	s = replace(s, mention_regex => "")
	s = replace(s, hashtag_regex => "")
	s = replace(s, url_regex => "")
	s = lowercase(s)
	
	sd = StringDocument(s)
	
	op = 0x00
	op |= strip_punctuation
	op |= strip_stopwords
	op |= strip_html_tags
	prepare!(sd, op)
	stem!(sd)
	table(ngrams(sd)), mentions, hashtags, urls
end

# ╔═╡ ef97e75c-1ff7-11eb-14a2-c74c20dda0e6
md"""
This strategy seems to work well. Let's make a copy of the data frame and start doing some analysis. 

P.S. I could have modified the original data frame but I would rather not do that because it will mess up the earlier part of this Pluto notebook.
"""

# ╔═╡ fd1a4328-1ff7-11eb-0b7f-2d62c0e9d0df
function extract_mentions(s)
	mention_regex = r"@\w+"
	return collect(x.match for x in eachmatch(mention_regex, s))
end;

# ╔═╡ 272dad1c-1ff8-11eb-0839-73226d473c08
function extract_hashtags(s)
	hashtag_regex = r"#\w+"
	return collect(x.match for x in eachmatch(hashtag_regex, s))
end;

# ╔═╡ 2ef4feb0-1ff8-11eb-05ef-ffff80db14ab
function extract_urls(s)
	url_regex = r"http[\w:/.]+"
	return collect(x.match for x in eachmatch(url_regex, s))
end;

# ╔═╡ b3b62480-1ff8-11eb-3ef9-7985882f19d4
function remove_extracted_text(s)
	mention_regex = r"@\w+"
	hashtag_regex = r"#\w+"
	url_regex = r"http[\w:/.]+"
	s = replace(s, mention_regex => "")
	s = replace(s, hashtag_regex => "")
	s = replace(s, url_regex => "")
	return s
end;

# ╔═╡ 457104ae-1ff8-11eb-04d8-536205dc3fc9
begin
	df2 = copy(df)
	transform!(df2, 
		:text => ByRow(extract_mentions) => :x_mentions,
		:text => ByRow(extract_hashtags) => :x_hashtags,
		:text => ByRow(extract_urls) => :x_urls,
		:text => ByRow(remove_extracted_text) => :x_text)
end;

# ╔═╡ 91e59d40-1ff8-11eb-0fab-df3732a771c2
table(df2[36, :])

# ╔═╡ d85f3010-1ff8-11eb-1d79-e338a3292aac
let	sd = StringDocument(lowercase(df2[36, :x_text]))
	op = 0x00
	op |= strip_punctuation
	op |= strip_stopwords
	op |= strip_html_tags
	prepare!(sd, op)
	stem!(sd)
	table(ngrams(sd))
end

# ╔═╡ 238efc32-1ff9-11eb-3887-17016268dfe2
function tweet_string_doc(s::AbstractString)
	sd = StringDocument(lowercase(s))
	op = 0x00
	op |= strip_punctuation
	op |= strip_stopwords
	op |= strip_html_tags
	prepare!(sd, op)
	# stem!(sd)
	return TextAnalysis.text(sd) == "" ? missing : sd
end;

# ╔═╡ 51471144-1ffb-11eb-37da-2595a74dbc64
function tweet_ngrams(sd::StringDocument, n = 1)
	try
		return NGramDocument(ngrams(sd, n))
	catch
		return missing
	end
end;

# ╔═╡ 904da938-1ffc-11eb-19ea-199dffd8d2c6
passmissing(f) = x -> ismissing(x) ? missing : f(x);

# ╔═╡ 433d3410-1ff9-11eb-235a-b147f848c2cf
begin
	transform!(df2, :x_text => ByRow(tweet_string_doc) => :x_string_doc)
	transform!(df2, :x_string_doc => ByRow(passmissing(tweet_ngrams)) => :x_ngrams)
	transform!(df2, :x_string_doc => ByRow(passmissing(x -> tweet_ngrams(x,2))) => :x_ngrams2)
	nothing
end

# ╔═╡ 9f723aa8-20c6-11eb-094e-d96421b204ee
md"""
What does it look like now?
"""

# ╔═╡ cd29da32-20c6-11eb-3e3b-adca4d3d31f4
table(df2[36, :])

# ╔═╡ da013d68-20c6-11eb-2166-973fae9b5d10
md"""
## Corpus analysis
"""

# ╔═╡ e3d9d5a0-20c6-11eb-00b6-51149f33ed3e
md"Here are the top 10 words"

# ╔═╡ cb06dd12-1ffa-11eb-3f46-cf0f210fce5e
let
	crps = Corpus(collect(skipmissing(df2.x_ngrams)))
	update_lexicon!(crps)
	lc = lexicon(crps)
	
	nts = [(count = v, word = k) for (k,v) in lc]
	sort!(nts, rev = true)
	
	DataFrame(nts[1:10])
end

# ╔═╡ ef40a858-20c6-11eb-3506-7f477e79cd74
md"Here are the top 2-grams"

# ╔═╡ 06fdc4d2-1ffd-11eb-04c9-0b4f549d9530
let
	crps = Corpus(collect(skipmissing(df2.x_ngrams2)))
	update_lexicon!(crps)
	lc = lexicon(crps)
	
	nts = [(count = v, word = k) for (k,v) in lc]
	sort!(nts, rev = true)
	
	DataFrame(nts[1:10])
end

# ╔═╡ 36831c92-20cb-11eb-0ea5-0fe7d0cecb1d
md"""
## Features
"""

# ╔═╡ 939eb5da-2139-11eb-019d-3b005ec5057c
md"Just trying out various functions in TextAnalysis.jl"

# ╔═╡ 54c37efe-2005-11eb-27b8-9ff7e99e6f86
begin
	crps = Corpus(collect(skipmissing(df2.x_string_doc)))
	update_lexicon!(crps)
end

# ╔═╡ 8ed4055a-20cb-11eb-0a25-bf2bc77326f6
DocumentTermMatrix(crps)

# ╔═╡ 92dcd85a-20cb-11eb-3272-d334eef04891
dtv(crps[1], lexicon(crps))

# ╔═╡ ce0d0944-20cb-11eb-1206-0558854e8c9e
hash_dtm(crps) 

# ╔═╡ e5f2ea18-20cb-11eb-2c11-ab6912382ea3
md"### TF (Term Frequency)"

# ╔═╡ f53166ee-20cb-11eb-26c0-f556545b65bb
tf(DocumentTermMatrix(crps))

# ╔═╡ 1c226af0-20cc-11eb-2e6c-a745f3fa11be
md"### TF-IDF (Term Frequency - Inverse Document Frequency)"

# ╔═╡ 32385066-20cc-11eb-1012-519181069bec
tf_idf(DocumentTermMatrix(crps))

# ╔═╡ 5c30f168-2124-11eb-3343-79f2765bd382
md"""
## Using Navie Bayes Classifier
"""

# ╔═╡ 3f30da46-2125-11eb-2ce7-7d44fb7229d0
md"The following sample code came from [this TextAnalysis.jl doc string](https://github.com/JuliaText/TextAnalysis.jl/blob/5f4edf9b39f8aa16c2aa47f109ae0d7971ceeef6/src/bayes.jl#L41-L63)."

# ╔═╡ 3694136e-2123-11eb-2fec-fd182545e5bb
let m = NaiveBayesClassifier([:spam, :non_spam])
	fit!(m, "this is spam", :spam)
	fit!(m, "this is not spam", :non_spam)
	predict(m, "is this a spam")
end

# ╔═╡ 38342910-2124-11eb-0092-0f667f375371
let m = NaiveBayesClassifier([:spam, :non_spam])
	fit!(m, "this is spam", :spam)
	fit!(m, "this is not spam", :non_spam)
	predict(m, "I'm not spam")
end

# ╔═╡ 6ebd952a-2124-11eb-30ee-2b80714ccb11
md"""
### Let's build our own classifier.

In our data frame, we already have a column `x_string_doc` with `StringDocuments` values. So we can just fit them to the classifier.
"""

# ╔═╡ 76ab55ba-2124-11eb-3999-71ce72708dd3
model = let classes = unique(df2.airline_sentiment)
	m = NaiveBayesClassifier(classes)
	for (sd, class) in zip(df2.x_string_doc , df2.airline_sentiment)
		fit!(m, sd, class) 
	end
	m
end;

# ╔═╡ 983678c6-2125-11eb-07a7-29321450d287
md"**Let's rock and roll!**"

# ╔═╡ cdcc1764-212c-11eb-1a24-cb5f0aa323a5
function test_model(predictor)
	df = DataFrame(text = [
		"whatever airline sucks!", 
		"i love @virginamerica service :-)", 
		"just ok", 
		"hello world"])
	
	df.analysis = predictor.(df.text)
	
	df.positive = getindex.(df.analysis, "positive")
	df.negative = getindex.(df.analysis, "negative")
	df.neutral = getindex.(df.analysis, "neutral")
	
	select!(df, Not(:analysis))
	
	return df
end;

# ╔═╡ 1df399ee-212d-11eb-2096-5189f234cb29
test_model(x -> predict(model, x))

# ╔═╡ c76ff982-2125-11eb-2d1b-c11a49051108
md"""
### Using features (words)

Since we have already generated 1-gram, I wonder if it could make the training faster.
"""

# ╔═╡ d739eefa-212e-11eb-19d9-df096c36079c
md"The `NaiveBayesClassifier` comes with another constructor that takes a vector of words to initize the underlying array. This can be found easily from the lexicon of the corpus."

# ╔═╡ d0640d26-2125-11eb-0181-9f19ccbd4791
lexicon(crps) |> keys |> length

# ╔═╡ f6a905a0-212e-11eb-376a-47ccd8f8b363
md"""
I realized that the TextAnalysis.jl package currently does not provide a `fit!` function for `NGramDocument`. Let's define a patch here just for fun!
"""

# ╔═╡ 49b7c392-212a-11eb-1334-21d96d1d8cf9
function TextAnalysis.fit!(c::NaiveBayesClassifier, ngd::NGramDocument, class)
	fs = ngrams(ngd)
    for k in keys(fs)
        k in c.dict || extend!(c, k)
    end
    fit!(c, TextAnalysis.features(fs, c.dict), class)
end

# ╔═╡ 1f88a1ba-212f-11eb-286c-8f82be0ce16e
md"Creating an ngram model is fairly straightforward."

# ╔═╡ fd5e0124-2125-11eb-1d70-f5316be8657f
ngrams_model = let 
	words = collect(keys(lexicon(crps)))
	classes = unique(df2.airline_sentiment)
	m = NaiveBayesClassifier(words, classes)
	for (ngd, class) in zip(df2.x_ngrams , df2.airline_sentiment)
		fit!(m, ngd, class) 
	end
	m
end;

# ╔═╡ 30414e62-212f-11eb-0a7b-f55ba0abce09
md"For testing, define a predictor function that takes any string (`x`) and it would determine its features and call the `predict` function with the model.)"

# ╔═╡ 910c5d08-212b-11eb-128e-4d112e08f255
function ngram_predictor(model)
	features(x) = TextAnalysis.features(ngrams(StringDocument(x)), ngrams_model.dict)
	return x -> predict(model, features(x))
end;

# ╔═╡ 58f866ce-212f-11eb-00e0-e5a8eb7013d9
md"**Let's rock and roll!**"

# ╔═╡ 8a71b98c-212d-11eb-3559-7bf3bc3a44bb
test_model(ngram_predictor(ngrams_model))

# ╔═╡ a64998a4-212e-11eb-1629-73b3134a1c8f
md"Let's compare with the result from using `StringDocument`. As expected, they are the same."

# ╔═╡ 90074a74-212d-11eb-18c1-4fd523eeec4e
test_model(x -> predict(model, x))

# ╔═╡ 02494270-2130-11eb-1342-390a8240d584
md"""
## Determining accuracy

What well does the Naive Bayes Classifier work?
"""

# ╔═╡ b4aa64ac-2136-11eb-142f-5dfb445e2e96
md"""
As the `predict` function returns a `Dict` object with the probabilities assigned to each class, we need to have choose the best option. Let's define a function for that.
"""

# ╔═╡ 54a76b44-2130-11eb-2680-211869c363da
function predict_and_choose(c::NaiveBayesClassifier, sd::StringDocument)
	val = predict(c, sd)
	return argmax(val)
end;

# ╔═╡ 0d66ccd4-2137-11eb-1f8e-2d03c16e29d9
md"Now, make prediction over all 14K tweets."

# ╔═╡ 0fbd02e8-2130-11eb-1f17-d77c0e80fa2a
df2.yhat = predict_and_choose.(Ref(model), df2.x_string_doc);

# ╔═╡ f27be3ec-2130-11eb-0114-33bf20368dba
hits = count(df2.airline_sentiment .== df2.yhat)

# ╔═╡ 55f555f6-2137-11eb-2312-99fbac7dad6e
misses = nrow(df2) - hits

# ╔═╡ 669e0fd8-2137-11eb-1eb4-01caefb25d55
wayoff = count(
			(df2.airline_sentiment .!== df2.yhat) .&
			(df2.airline_sentiment .!== "neutral") .&
			(df2.yhat .!== "neutral"))

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

# ╔═╡ 417a6bba-212e-11eb-047d-e5b9891b7873
md"""
## To-do's

Things that I'd like to do but don't have time at the moment.

1. Try n-gram with 1 and 2 words and see if it predicts better. However, this will explode the number of features a lot. Just doing 1-gram already gives 12K features.

2. Stemming will probably reduce the size. Not sure how it affects predictive power.
"""

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
# ╠═3fab77c4-1ff2-11eb-02bf-ed577baacef7
# ╠═4797ae3c-1ff2-11eb-3f31-f32f365ea0ee
# ╠═bec8eb92-1ff2-11eb-033d-ff5d3be4df0a
# ╠═2fa37940-1ff3-11eb-01ef-813c561d5679
# ╠═747a707c-1ff4-11eb-2681-6b6787174d0e
# ╠═5248a178-1ff3-11eb-383e-993d9e6287d9
# ╟─5ca1e714-1f8b-11eb-30fd-dbdd6e574805
# ╟─097e9b2c-1ff4-11eb-255d-1d0816f3ddee
# ╠═74192f10-1fe0-11eb-3eea-1d9cbc0c4c3b
# ╟─32974c34-1ff4-11eb-18ea-67999c24906a
# ╠═e97bc70a-2002-11eb-3f57-ab1838225b0e
# ╟─db9272f6-1ff3-11eb-12e7-0fa2c8837975
# ╠═cbcac7e6-1fe0-11eb-0820-9f4c6f3fa51f
# ╟─dd832f9c-1f8a-11eb-001b-3d8be911d80a
# ╟─b231f13c-1ff0-11eb-278d-43db738ca57f
# ╟─89533ac8-1ff5-11eb-0838-21d9d514ab8a
# ╠═04c821e0-1fe0-11eb-3a1b-670c5ff35a90
# ╠═1c0b8dea-1fe0-11eb-19a2-49a1af2ed48c
# ╟─b636fc32-1ff5-11eb-048e-cbc4a77c5f75
# ╠═3df9b88c-1fe0-11eb-1032-31d3baa1a91b
# ╠═6cb54136-1f8c-11eb-21d0-1542ddd12437
# ╟─cd6619b0-1ff5-11eb-19f5-092f2f06b172
# ╠═3247ecdc-1ff6-11eb-2a79-85b10f64a651
# ╟─39546492-1ff6-11eb-1848-37e0d8420226
# ╟─8a3e8bbc-1ff6-11eb-2a12-c3fba022267b
# ╠═9e3133ea-1ff6-11eb-3857-4f2dd7436ccb
# ╟─ef97e75c-1ff7-11eb-14a2-c74c20dda0e6
# ╠═fd1a4328-1ff7-11eb-0b7f-2d62c0e9d0df
# ╠═272dad1c-1ff8-11eb-0839-73226d473c08
# ╠═2ef4feb0-1ff8-11eb-05ef-ffff80db14ab
# ╠═b3b62480-1ff8-11eb-3ef9-7985882f19d4
# ╠═457104ae-1ff8-11eb-04d8-536205dc3fc9
# ╠═91e59d40-1ff8-11eb-0fab-df3732a771c2
# ╠═d85f3010-1ff8-11eb-1d79-e338a3292aac
# ╠═238efc32-1ff9-11eb-3887-17016268dfe2
# ╠═51471144-1ffb-11eb-37da-2595a74dbc64
# ╠═904da938-1ffc-11eb-19ea-199dffd8d2c6
# ╠═433d3410-1ff9-11eb-235a-b147f848c2cf
# ╟─9f723aa8-20c6-11eb-094e-d96421b204ee
# ╠═cd29da32-20c6-11eb-3e3b-adca4d3d31f4
# ╟─da013d68-20c6-11eb-2166-973fae9b5d10
# ╟─e3d9d5a0-20c6-11eb-00b6-51149f33ed3e
# ╠═cb06dd12-1ffa-11eb-3f46-cf0f210fce5e
# ╟─ef40a858-20c6-11eb-3506-7f477e79cd74
# ╠═06fdc4d2-1ffd-11eb-04c9-0b4f549d9530
# ╟─36831c92-20cb-11eb-0ea5-0fe7d0cecb1d
# ╟─939eb5da-2139-11eb-019d-3b005ec5057c
# ╠═54c37efe-2005-11eb-27b8-9ff7e99e6f86
# ╠═8ed4055a-20cb-11eb-0a25-bf2bc77326f6
# ╠═92dcd85a-20cb-11eb-3272-d334eef04891
# ╠═ce0d0944-20cb-11eb-1206-0558854e8c9e
# ╟─e5f2ea18-20cb-11eb-2c11-ab6912382ea3
# ╠═f53166ee-20cb-11eb-26c0-f556545b65bb
# ╟─1c226af0-20cc-11eb-2e6c-a745f3fa11be
# ╠═32385066-20cc-11eb-1012-519181069bec
# ╟─5c30f168-2124-11eb-3343-79f2765bd382
# ╟─3f30da46-2125-11eb-2ce7-7d44fb7229d0
# ╠═2b71ba0e-2123-11eb-0f43-bb72fb34208a
# ╠═3694136e-2123-11eb-2fec-fd182545e5bb
# ╠═38342910-2124-11eb-0092-0f667f375371
# ╟─6ebd952a-2124-11eb-30ee-2b80714ccb11
# ╠═76ab55ba-2124-11eb-3999-71ce72708dd3
# ╟─983678c6-2125-11eb-07a7-29321450d287
# ╠═cdcc1764-212c-11eb-1a24-cb5f0aa323a5
# ╠═1df399ee-212d-11eb-2096-5189f234cb29
# ╟─c76ff982-2125-11eb-2d1b-c11a49051108
# ╟─d739eefa-212e-11eb-19d9-df096c36079c
# ╠═d0640d26-2125-11eb-0181-9f19ccbd4791
# ╟─f6a905a0-212e-11eb-376a-47ccd8f8b363
# ╠═49b7c392-212a-11eb-1334-21d96d1d8cf9
# ╟─1f88a1ba-212f-11eb-286c-8f82be0ce16e
# ╠═fd5e0124-2125-11eb-1d70-f5316be8657f
# ╟─30414e62-212f-11eb-0a7b-f55ba0abce09
# ╠═910c5d08-212b-11eb-128e-4d112e08f255
# ╟─58f866ce-212f-11eb-00e0-e5a8eb7013d9
# ╠═8a71b98c-212d-11eb-3559-7bf3bc3a44bb
# ╟─a64998a4-212e-11eb-1629-73b3134a1c8f
# ╠═90074a74-212d-11eb-18c1-4fd523eeec4e
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
# ╟─417a6bba-212e-11eb-047d-e5b9891b7873
