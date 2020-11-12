### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ 845202ce-23d9-11eb-085e-ad2c26fd531c
begin
	using TextAnalysis
	using CSV
	using DataFrames
end

# ╔═╡ cac12010-23d8-11eb-0321-3fc66858fc84
md"# Authorship Analysis using TextAnalysis.jl

In this notebook we will implement an authorship analysis technique that uses n-gram frequencies for identifying authors. This technique is described in [Grieve 2018](https://research.birmingham.ac.uk/portal/files/53402456/Bixby_PREPRINT.pdf) The technique counts token and character n-grams and compares them to a reference corpus. It counts the number of overlapping n-grams in the reference corpus and identifies the author based on which reference corpus contains more of the n-grams from the text."

# ╔═╡ f400d820-23d9-11eb-091a-898d693456ad
md"First off, the necessary imports are made."

# ╔═╡ c9d140dc-2463-11eb-06a2-2d0dcb8dbed2
md"Our data will be from [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification/overview) Kaggle contest. The data comes in the form of a CSV file which contains sentences attributed to one of three authors. The data was manualy cleaned to remove rows that were corropted as well as making sure that each author had approximately equal word counts.

The data is read into a dataframe after which we iterate over the df and turn each sentence into a StringDocument which is then accumalted into a Corpus for each author."

# ╔═╡ 68255d02-23da-11eb-3232-613c0e1b0ee2
referenceData = CSV.read("train.csv", DataFrame);

# ╔═╡ c8b1e874-23e0-11eb-1aad-995c03f040ef
begin
	MWS = GenericDocument[]
	EAP = GenericDocument[]
	HPL = GenericDocument[]
	
	for line in eachrow(referenceData)
		if line.author == "MWS"
			push!(MWS, StringDocument(line.text))
		elseif line.author == "EAP"
			push!(EAP, StringDocument(line.text))
		else
			push!(HPL, StringDocument(line.text))
		end
	end
	
	MWS_corpus = Corpus(MWS)
	EAP_corpus = Corpus(EAP)
	HPL_corpus = Corpus(HPL);
end

# ╔═╡ 60390844-2472-11eb-010b-d155abe74997
md"The following code snippet allows one to retrieve the word count of a corpus. The word counts are as follows:
- MWS_corpus: 147914
- EAP_corpus: 147911
- HPL_corpus: 147916"

# ╔═╡ 7b9ce77e-23e4-11eb-39d4-03e26b33cad7
begin
	wordCount = 0
	for doc in HPL_corpus
		global wordCount += length(ngrams(doc, 1))
	end
end

# ╔═╡ 43a58408-246f-11eb-29ab-91cc00373c2a
wordCount

# ╔═╡ 96f5d60a-248b-11eb-2795-a51a9fbf3fc0
md"Now that we have our corpora it is time to extract the ngrams that are needed for the analysis. The original paper used both character ngrams and token ngrams. Due to TextAnalysis.jl's functionality I will be restricting this analysis to only using token ngrams."

# ╔═╡ 95691ad4-2472-11eb-346e-07a9f5dc8f6c
begin
	MWS_ngrams = []
	HPL_ngrams = []
	EAP_ngrams = []
	for i in 1:4
		push!(MWS_ngrams, map(x -> ngrams(x,i), MWS_corpus))
		push!(HPL_ngrams, map(x -> ngrams(x,i), HPL_corpus))
		push!(EAP_ngrams, map(x -> ngrams(x,i), EAP_corpus))
	end
end

# ╔═╡ e207949e-2526-11eb-3ae4-0712b7082681
md"We also retrieve the token ngrams from the files we want to analyze and classify."

# ╔═╡ 62443a4c-2525-11eb-0e70-393860da7b4f
begin
	# first we load our files from memory
	text1 = FileDocument("./MWS.txt")
	text2 = FileDocument("./HPL.txt")
	text3 = FileDocument("./EAP.txt")
	
	# this is the same as retrieveing ngrams for the reference corpora
	text1_ngrams = []
	text2_ngrams = []
	text3_ngrams = []
	for i in 1:4
		push!(text1_ngrams, ngrams(text1, i))
		push!(text2_ngrams, ngrams(text2, i))
		push!(text3_ngrams, ngrams(text3, i))
	end
end

# ╔═╡ 32909f1e-2527-11eb-0efd-6d81943df28e
md"Finally we will a script that calculates the intersection of the ngrams from the text and the three reference corpora. To try different files edit the second for loop."

# ╔═╡ 5d1320ee-2479-11eb-3664-c33f5ace1ba7
begin
	MWS_count = 0
	HPL_count = 0
	EAP_count = 0
	for i in 1:4
		for j in text1_ngrams[i]
			for k in MWS_ngrams[i]
				if haskey(k, j[1])
					global MWS_count += 1
					break
				end
			end
			for k in HPL_ngrams[i]
				if haskey(k, j[1])
					global HPL_count += 1
					break
				end
			end
			for k in EAP_ngrams[i]
				if haskey(k, j[1])
					global EAP_count += 1
					break
				end
			end
		end
	end
end

# ╔═╡ 16a5c100-252a-11eb-1258-b769d50904d3
md"The following block will output the author as identified by this script.
The correct authors should be as follows:
- text1 = MWS
- text2 = HPL
- text3 = EAP"

# ╔═╡ df84c6ee-247a-11eb-24f9-e92653c334bb
begin
	if MWS_count > HPL_count && MWS_count > EAP_count
		"MWS"
	elseif HPL_count > MWS_count && HPL_count > EAP_count
		"HPL"
	else
		"EAP"
	end
end

# ╔═╡ 34de486c-252b-11eb-2071-f12f0f445d9f
md"## Conclusion
While this is a very watered down analysis of what the paper presents it is a good starting point and shows that this technique is viable as it correctly predicts the authors of our random test data. One way to improve the current analysis is to add character ngrams as the paper does.

All in all this was a fun project to learn more about Julia and its NLP cabalities. This project only scratches the surface of what Julia has to offer given that the TextAnalysis.jl package offers things such as part-of-speech(POS) tagging, tf-idf measures, a sentiment analysis model, and much more."

# ╔═╡ Cell order:
# ╟─cac12010-23d8-11eb-0321-3fc66858fc84
# ╟─f400d820-23d9-11eb-091a-898d693456ad
# ╠═845202ce-23d9-11eb-085e-ad2c26fd531c
# ╟─c9d140dc-2463-11eb-06a2-2d0dcb8dbed2
# ╠═68255d02-23da-11eb-3232-613c0e1b0ee2
# ╠═c8b1e874-23e0-11eb-1aad-995c03f040ef
# ╟─60390844-2472-11eb-010b-d155abe74997
# ╠═7b9ce77e-23e4-11eb-39d4-03e26b33cad7
# ╠═43a58408-246f-11eb-29ab-91cc00373c2a
# ╟─96f5d60a-248b-11eb-2795-a51a9fbf3fc0
# ╠═95691ad4-2472-11eb-346e-07a9f5dc8f6c
# ╟─e207949e-2526-11eb-3ae4-0712b7082681
# ╠═62443a4c-2525-11eb-0e70-393860da7b4f
# ╟─32909f1e-2527-11eb-0efd-6d81943df28e
# ╠═5d1320ee-2479-11eb-3664-c33f5ace1ba7
# ╟─16a5c100-252a-11eb-1258-b769d50904d3
# ╠═df84c6ee-247a-11eb-24f9-e92653c334bb
# ╟─34de486c-252b-11eb-2071-f12f0f445d9f
