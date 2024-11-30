module utils
	
	using FCSeqTools
	using PyCall
	using PyPlot
	using FCSeqTools
	using Statistics, LinearAlgebra

	export id, J2square_format, read_fasta, read_secondary_strucuture, one_hot_encoding, zerosum_gauge, frobenius_norm, plot_contact_map, read_fitness, plot_DMS_prediction, context_dependent_entropy, context_independent_entropy, read_sequences_and_labels

   	id(i, a, q) = (i .- 1).*q .+ a

	function J2square_format(J, q)
	L = size(J, 1)
	J_square = zeros(Float32, L*q, L*q)
	for i in 1:L, j in i+1:L
	J_square[id(i, 1:q, q), id(j, 1:q, q)] = reshape(J[i, j, :], (q, q))
	J_square[id(j, 1:q, q), id(i, 1:q, q)] = reshape(J[i, j, :], (q, q))'
	end
	return J_square
	end
    
	read_fasta(fasta_path) = do_number_matrix_rna(do_letter_matrix(fasta_path), 1)

	function read_secondary_strucuture(file_path)
		contact_matrix = []
		open(file_path, "r") do file
			readline(file)
			(_, contact_matrix) = dot_bracket_to_ss_matrix(readline(file))
		end
		return contact_matrix
	end
	
	function one_hot_encoding(V, Nq)
	  V = V'
	  Nv, Ns = size(V)
	  oneHotV = BitArray(undef, Nq, Nv, Ns)
	  all_v = collect(1:Nq)
	  for i_s in 1:Ns, i_v in 1:Nv
	      oneHotV[:, i_v, i_s] = (all_v .== V[i_v, i_s])
	  end
	  oneHotV = reshape(oneHotV, (Nq*Nv, Ns))'
	  return Float32.(oneHotV)
	end
	
	
	
	
	function zerosum_gauge(J, h, q, L)
	  J_gauge = zeros(Float32, size(J))
	  for i in 1:L, j in 1:L
	    Jij_mat = reshape(J[i, j, :], (q, q))
	    J_gauge[i, j, :] .= J[i, j, :] .+ mean(J[i, j, :]) .- vec(mean(Jij_mat, dims=1) .+ mean(Jij_mat, dims=2))
	  end
	 
	  for i in 1:L
	    for a in 1:q
	      for j in 1:L
		Jij_mat = reshape(J[i, j, :], (q, q))
		h[id(i, a, q)] .- Jij_mat[a, :]
	      end
	    end
	  end

	  return J, h
	end



	function frobenius_norm(J, q, L)
	  J_contact = zeros(L, L)
	  for i in 1:L, j in i+1:L
		J_contact[i, j] = norm(J[i, j, :], 2)
	  end
	  return J_contact
	end
	
	
	function plot_contact_map(J_contact, contact_matrix_1, threshold)

	  predicted_contacts = (J_contact .>= threshold)

	  true_predicted = (predicted_contacts .* contact_matrix_1)
	  true_predicted_xy = findall(x -> x .!= 0, true_predicted)
	  x_true = getindex.(true_predicted_xy, 1)
	  y_true = getindex.(true_predicted_xy, 2)

	  false_predicted_xy = findall(x -> x .!= 0, (predicted_contacts .!= true_predicted))
	  x_wrong = getindex.(false_predicted_xy, 1)
	  y_wrong = getindex.(false_predicted_xy, 2)

	  contacts = findall(x -> x .> 0, contact_matrix_1)
	  x = getindex.(contacts, 1)
	  y = getindex.(contacts, 2)

	  plt.figure(figsize=(7, 7))
	  plt.xlim(0, 31)
	  plt.ylim(0, 31)
	  plt.scatter(x_true, y_true, label="true predicted", color="green")
	  plt.scatter(x_wrong, y_wrong, label="false predicted", color="red")
	  plt.scatter(y, x, label="contacts", color="blue")

	  plt.title("Contact Map")
	  plt.xlabel("Nucleotide Positions")
	  plt.ylabel("Nucleotide Positions")
	  plt.grid()
	  plt.legend()
	  plt.show()
	end

	function read_fitness(file_name::String)
	    vec = Float64[]
	    open(file_name, "r") do file
		for line in eachline(file)
		    push!(vec, parse(Float64, line))
		end
	    end
	    return vec
	end

	function plot_DMS_prediction(fitness, score_vector)
	    plt.scatter(fitness, score_vector, color="blue", edgecolors="black")
	    ro = cor(fitness, score_vector)
	    plt.grid()
	    plt.title("DMS - Pearson coeff: $(ro)")
	    plt.xlabel("Fitness")
	    plt.ylabel("Score")
	    plt.show()
	end
	
	function context_dependent_entropy(sequence, L, q, J, h)
	  entropy = zeros(Float64, L)
	  for i in 1:L 
	    ene = zeros(Float64, q-1)
	    for a in 1:q-1
	      seq2 = copy(sequence)
	      seq2[i] = Int(a) 
	      #println(sequence)
	      ene[a] = sequences_energy(seq2', q, h, J)[1]
	    end
	    #println("ene: ", ene)
	    p = exp.(- ene)
	    #println("p: ", p)
	    p =  p ./ sum(p)
	    #println("p: ", p)

	    entropy[i] = -sum( p .* log.(p))
	  end
	  return entropy
	end


	function context_independent_entropy(data, q)
	  return site_entropy_vector(data, q, 0.1, 1)
	end

	function read_sequences_and_labels(file_path::String)
		labels = Int[]
		# Open the file for reading
		open(file_path, "r") do file
			# Read each line
			for line in eachline(file)
				# Check if the line is a label line (starts with ">")
				if startswith(line, ">")
					# Extract the label (last character after the last underscore)
					label = parse(Int, last(split(line, "_")))
					push!(labels, label)
				end
			end
		end
		sequences = read_fasta(file_path);
		return sequences, labels
	end
    
end
