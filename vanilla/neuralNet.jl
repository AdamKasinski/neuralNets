using Match, Statistics, Distributions, Random

@enum ActivationFunctions begin
    ReLu
    Sigmoid
end

struct Layer
    weights::Matrix{Float32}
    biases::Array{Float32}
end

struct Network #TODO add weights to Network structure - weights should not be in layers - it is not intuitive  
    layers::Array{Layer}
end

function createLayer(neuronsInTheLayer,neuronsInPreviousLayer)
    d = Normal()
    weights::Matrix{Float32} = rand(d,(neuronsInPreviousLayer,neuronsInTheLayer))
    biases::Array{Float32}  = rand(d,neuronsInTheLayer)
    return Layer(weights,biases)
end

function createNetwork(numberOfLayers::Int,neuronsInLayers::Array{Int})
    layers = Array{Union{Layer,Nothing}}(nothing,numberOfLayers)
    for layerNumber in 1:numberOfLayers
        previousLayerNumber = layerNumber == 1 ?  1 : layerNumber-1
        layers[layerNumber] = createLayer(neuronsInLayers[layerNumber],neuronsInLayers[previousLayerNumber])
    end 
    return Network(layers)
end

function sigmoid(x::Float32)
    return 1/(1+ℯ^(-x))
end

function derivativeSigmoid(x::Float32)
    return x*(1-x)
end

function relu(x::Float32)
    return max(0,x)
end

function calculateDeltaOutput(results,expectedResults)
    errors::Array{Float32} = results - expectedResults
    return errors.*derivativeSigmoid(results)
end

function calculateDeltaHidden(currentDelta, weights,results)
    error = currentDelta*weights
    return error.*derivativeSigmoid(results)
end

function forward(inputs::Array{Float32}, layer::Layer ,fun::ActivationFunctions) 
    z = layer.weights'*inputs.+layer.biases
    d = 0
    return @match fun begin
        $ReLu => relu.(z)
        $Sigmoid => sigmoid.(z)
    end
end

function evaluate(results::Array{Float32}, expectedResults::Array{Float32})
    return sum((results - expectedResults).^2)
end

function forwardPropagation(input, network::Network, fun::ActivationFunctions)
    inputs = []
    for layer in network.layers
        input = forward(input,layer,fun)
        push!(inputs,input)
    end
    return inputs
end


function backpropagation(network::Network,results::Array{Float32}, expected_results::Array{Float32})
    
    δ::Array{Float32} = []
    push!(δ,calculateDeltaOutput(results[end], expected_results))

    for layerNumber in reverse(2:length(network.layers))
        delta = calculateDeltaHidden(δ[-1],network.layers[layerNumber+1].weights,results[layerNumber])
        push!(δ,delta)
    end
    
    return reverse(δ)
end

function updateWeights(network,learningRate, δ, results)
    for layerNumber in 1:length(network.layers)
        network.layers[layerNumber].weights += learningRate*(results[layerNumber]'*δ[layerNumber])
        network.layers[layerNumber].biases += learningRate*sum(δ[layerNumber])
    end
end