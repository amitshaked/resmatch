local network = require 'networks/network'

local L2dist, parent = torch.class('nn.L2dist', 'nn.Module')

function L2dist:__init()
   parent.__init(self)
   self.gradInput = torch.CudaTensor()
   self.tmp = torch.CudaTensor()
   self.diff = torch.CudaTensor()
   self.outExpand = torch.CudaTensor()
   self.ones = torch.CudaTensor()
   self.output = torch.CudaTensor()
end


function L2dist:updateOutput(input)
   local input_L, input_R = network.sliceInput(input)
   self.diff:resizeAs(input_L)
   self.diff:zero()
   self.diff:add(input_L, -1, input_R):pow(2)
   self.output:sum(self.diff, 2)
   self.output:pow(1./2)
   return self.output
end


function L2dist:updateGradInput(input, gradOutput)
   -- input[2*i-1]: the i'th left patch in the batch
   -- input[2*i]: the i'th right patch in the batch
   -- self.output = ||input_L - input_R ||_2
   -- the gradInput should be the derivative of 2-norm:
   --    d/dx_k(||x - y||_2) = (x_k -y_k) * x_k' / (||x-y||_2)

   local input_L, input_R = network.sliceInput(input)
   self.gradInput:resizeAs(input)

   local gradInput_L, gradInput_R = network.sliceInput(self.gradInput)
   gradInput_L:add(input_L, -1, input_R)

   self.outExpand:resizeAs(self.output)
   self.outExpand:copy(self.output)

   self.outExpand:add(1.0e-6) -- Prevent divide by zero errors
   self.outExpand:pow(-1)
   
   gradInput_L:cmul(self.outExpand:expandAs(gradInput_L))
   self.grad = self.grad or gradOutput.new()
   self.ones = self.ones or gradOutput.new()
   self.grad:resize(input_L:size(1), input_L:size(2)):zero()
   self.ones:resize(input_L:size(2)):fill(1)
   self.grad:addr(gradOutput:squeeze(), self.ones)
   gradInput_L:cmul(self.grad)
   gradInput_R:zero():add(-1,gradInput_L)
   return self.gradInput
end

function L2dist:computeMatchingCost(input_L, input_R, output_L, output_R)
   -- computes matching cost for all pixels and all disperities at ones
   adcensus.L2dist(input_L, input_R, output_L, output_R)
end
