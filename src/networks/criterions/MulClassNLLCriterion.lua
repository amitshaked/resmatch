local MulClassNLLCriterion, parent = torch.class(
'nn.MulClassNLLCriterion',
'nn.Criterion'
)

function MulClassNLLCriterion:__init(gt_weight)
   parent.__init(self)

   if gt_weight then
      gt_weight:div(gt_weight:sum())
      self.gt_weight = gt_weight
   else
      self.gt_weight = torch.ones(1)
   end

   assert(self.gt_weight:nElement() % 2 == 0, 'nElement of gt_weight should be even')
   self.half_width = (self.gt_weight:nElement())/ 2
end




function MulClassNLLCriterion:__len()
   return 0
end

function MulClassNLLCriterion:updateOutput(input, target)
   assert(type(target) ~= 'number', 'target should be a tensor')

   if target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   -- has dimension for batch-size
   assert(input:dim() == 2 and target:dim() == 2, 'input should be 2D')
   assert(target:size(2) == 1, string.format('only support 1 gt locaton, got: %d', target:size(2)))
   self.output = 0
   for i = 1,input:size(1) do
      local t = math.floor(target[i][1])
      local s = math.max(1, t-self.half_width + 1)
      local e = math.min(input:size(2), t+self.half_width)

      local probs= input[{i,{s,e}}]
      local weights = self.gt_weight[{{self.half_width-(t-s), self.half_width+ (e-t)}}]

      self.output = self.output - torch.cmul(probs, weights):sum()
   end
   self.output = self.output / input:nElement()
   return self.output
end

function MulClassNLLCriterion:updateGradInput(input, target)
   assert(type(target) ~= 'number', 'target should be a tensor')

   if target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   assert(input:dim() == 2, 'input should be 2D')
   self.gradInput:resizeAs(input):zero()

   for i = 1,input:size(1) do
      local t = math.floor(target[i][1])
      local s = math.max(1, t-self.half_width +1)
      local e = math.min(input:size(2), t+self.half_width)

      self.gradInput[{i,{s,e}}]:copy(self.gt_weight[{{self.half_width-(t-s), self.half_width+e-t}}]):mul(-1)
   end

   self.gradInput:div(target:nElement())
   return self.gradInput
end
