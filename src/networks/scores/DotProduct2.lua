local network = require('networks/network')
local DotProduct2, parent = torch.class('nn.DotProduct2', 'nn.Module')

function DotProduct2:__init()
   parent.__init(self)
   self.gradInput = torch.CudaTensor()
   self.tmp = torch.CudaTensor()
   self.output = torch.CudaTensor()
end

function DotProduct2:updateOutput(input)
   local input_L, input_R = network.sliceInput(input)
   self.tmp:resizeAs(input_L)
   self.tmp:cmul(input_L, input_R)
   self.output:sum(self.tmp, 2)
   return self.output
end

function DotProduct2:updateGradInput(input, gradOutput)
   gradOutput:cuda()
   input:cuda()
   self.gradInput:resizeAs(input)
   local input_L, input_R = network.sliceInput(input)
   local gradInput_L, gradInput_R = network.sliceInput(self.gradInput)
   gradInput_L:cmul(input_R, gradOutput:expandAs(input_R):cuda())
   gradInput_R:cmul(input_L, gradOutput:expandAs(input_L):cuda())
   return self.gradInput
end

function DotProduct2:computeMatchingCost(input_L, input_R, output_L, output_R)
   adcensus.StereoJoin(input_L, input_R, output_L, output_R)
end

