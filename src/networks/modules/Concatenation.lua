local Concatenation, parent = torch.class('nn.Concatenation', 'nn.Module')

function Concatenation:__init()
   parent.__init(self)
end

function Concatenation:updateOutput(input)

	self.output = input:view(input:size(1)/2, input:size(2)*2, input:size(3), input:size(4))
   return self.output
end


function Concatenation:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:view(input:size())
	return self.gradInput
end
