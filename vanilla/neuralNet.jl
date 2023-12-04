using Match, Statistics, Distributions, Random

@enum ActivationFunctions begin
    ReLu
    Sigmoid
end


struct Layer
    weights::Matrix{Float32}
    biases::Array{Float32}
end

struct Network
    layers::Array{Layer}
end

function sigmoid(x::Float32)
    return 1/(1+ℯ^(-x))
end

function relu(x::Float32)
    return max(0,x)
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

function forward(inputs::Array{Float32}, layer::Layer ,fun::ActivationFunctions) 
    z = layer.weights'*inputs.+layer.biases
    d = 0
    return @match fun begin
        $ReLu => relu.(z)
        $Sigmoid => sigmoid.(z)
    end
end

function evaluate(results::Float32, expectedResults::Float32)
    return sum((results - expectedResults).^2)
end

function forwardpropagation(input, network::Network, fun::ActivationFunctions)
    for layer in network.layers
        input = forward(input,layer,fun)
    end
    return input
end


function backpropagation()
    pass
end