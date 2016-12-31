local SpatialConvolution1_fw, parent = torch.class('nn.SpatialConvolution1_fw', 'nn.Module')

function SpatialConvolution1_fw:__init(inputSize, outputSize, free)
   parent.__init(self)
   self.weight = torch.CudaTensor(outputSize, inputSize)
   self.bias = torch.CudaTensor(1, outputSize, 1, 1)
   self.output:cuda()
	self.free = free -- free memory
end

function SpatialConvolution1_fw:updateOutput(input)
   local num_ex = input:size(1)
   local fm_in = input:size(2)
   local h = input:size(3)
   local w = input:size(4)
   local fm_out = self.weight:size(1)

   input:resize(num_ex, fm_in, h * w)
   self.output:resize(num_ex, fm_out, h * w)
   for i = 1,num_ex do
      self.output[i]:addmm(0, 1, self.weight, input[i])
   end
   input:resize(num_ex, fm_in, h, w)
   self.output:resize(num_ex, fm_out, h, w)

   self.output:add(self.bias:expandAs(self.output))

   -- Free memory
	if self.free then
   	input:storage():resize(0)
	end
   return self.output
end
