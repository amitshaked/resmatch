require('networks/modules/Concatenation')
require('networks/modules/SpatialConvolution1_fw')

local network = {}
local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function network.clean(model)
   return deepCopy(model):float():clearState()
end


local function convInit(model, name)
   for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if cudnn.version >= 4000 then
         v.bias = nil
         v.gradBias = nil
      else
         v.bias:zero()
      end
   end
end

local function bNInit(model, name)
   for k,v in pairs(model:findModules(name)) do
      v.weight:fill(1)
      v.bias:zero()
   end
end

local function linearInit(model, name)
   for k,v in pairs(model:findModules(name)) do
      v.bias:zero()
   end
end

function network.getWindowSize(net, ws)
   ws = ws or 1

   for i = 1,#net.modules do
      local module = net:get(i)
      if torch.typename(module) == 'cudnn.SpatialConvolution' then
         ws = ws + module.kW - 1 - module.padW - module.padH
      end
      if module.modules then
         ws = network.getWindowSize(module, ws)
      end
   end
   return ws
end

function network.init(net)
   convInit(net, 'cudnn.SpatialConvolution')
   convInit(net, 'nn.SpatialConvolution')
   bNInit(net, 'cudnn.SpatialBatchNormalization')
   bNInit(net, 'nn.SpatialBatchNormalization')
   linearInit(net, 'nn.Linear')
end

function network.fixBorder(vol, direction, ws)
   local n = (ws - 1) / 2
   for i=1,n do
      vol[{{},{},{},direction * i}]:copy(vol[{{},{},{},direction * (n + 1)}])
   end
end

local function padConvs(module)
   -- Pads the convolutional layers to maintain the image resolution
   for i = 1,#module.modules do
      local m = module:get(i)
      if torch.typename(m) == 'cudnn.SpatialConvolution' then
         m.dW = 1
         m.dH = 1
         if m.kW > 1 then
            m.padW = (m.kW - 1) / 2
         end
         if m.kH > 1 then
            m.padH = (m.kH - 1) / 2
         end
      elseif m.modules then
         padConvs(m)
      end
   end
end

function network.getTestNetwork(model)
   -- Replace the model with fully-convolutional network
   -- with the same weights, and pad it to maintain resolution

   local testModel = model:clone('weight', 'bias')

   -- replace linear with 1X1 conv
   local nodes, containers = testModel:findModules('nn.Linear')
   for i = 1, #nodes do
      for j = 1, #(containers[i].modules) do
         if containers[i].modules[j] == nodes[i] then

            local w = nodes[i].weight
            local b = nodes[i].bias
            local conv = nn.SpatialConvolution1_fw(w:size(2), w:size(1)):cuda()
            conv.weight:copy(w)
            conv.bias:copy(b)
            -- Replace with a new instance
            containers[i].modules[j] = conv
         end
      end
   end

   -- replace reshape with concatenation
   nodes, containers = testModel:findModules('nn.Reshape')
   for i = 1, #nodes do
      for j = 1, #(containers[i].modules) do
         if containers[i].modules[j] == nodes[i] then
            -- Replace with a new instance
            containers[i].modules[j] = nn.Concatenation():cuda()
         end
      end
   end

   -- pad convolutions
   padConvs(testModel)

   -- switch to evalutation mode
   testModel:evaluate()


   return testModel
end

function network.forwardFree(net, input)
   -- Forwards the network w.r.t input module by module
   -- while cleaning previous modules state
   local currentOutput = input
   for i=1, #net.modules do
      local m  = net.modules[i]
      local nextOutput
      if torch.typename(m) == 'nn.Sequential' then
         nextOutput = network.forwardFree(m, currentOutput)
         currentOutput = nextOutput:clone()
      elseif torch.typename(m) == 'nn.ConcatTable' or torch.typename(m) == 'nn.ParallelTable' then
         nextOutput = m:forward(currentOutput)
         currentOutput = {}
         currentOutput[1] = nextOutput[1]:clone()
         currentOutput[2] = nextOutput[2]:clone()
      else
         nextOutput = m:updateOutput(currentOutput)
         currentOutput = nextOutput:clone()
      end
      m:apply(
      function(mod)
         mod:clearState()
      end
      )

      collectgarbage()
   end

   return currentOutput
end

function network.sliceInput(input)
   local sizes = torch.LongStorage{input:size(1) / 2, input:size(2), input:size(3), input:size(4)}
   local strides = torch.LongStorage{input:stride(1) * 2, input:stride(2), input:stride(3), input:stride(4)}

   local input_L = torch.CudaTensor(input:storage(), 1, sizes, strides)
   local input_R = torch.CudaTensor(input:storage(), input:stride(1) + 1, sizes, strides)

   return input_L, input_R
end


Normalization = nn.Normalize2
Activation = cudnn.ReLU
Convolution = cudnn.SpatialConvolution
Avg = cudnn.SpatialAveragePooling
Max = nn.SpatialMaxPooling

return network
