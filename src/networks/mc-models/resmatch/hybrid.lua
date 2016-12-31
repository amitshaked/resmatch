local network = require 'networks/network'
require 'networks/mc-models/resmatch/acrt'
require('networks/criterions/Margin2')
require('networks/scores/DotProduct2')
require('networks/modules/Normalize2')

local ResmatchHybrid, parent = torch.class('ResmatchHybrid','ResmatchAcrt')

function ResmatchHybrid:__init(self, opt, dataset)

   parent.__init(parent, self, opt, dataset)
   local bce = nn.BCECriterion2()
   local margin = nn.Margin2()

   -- parallel criterion with repeated target
   self.criterion = nn.ParallelCriterion(true)
      :add(bce, 0.8)
      :add(margin, 0.2)
      :cuda()

   self.params.arch= {{1,2},{1,2},{1,2},{1,2},{1,2}}
end

function ResmatchHybrid:getDecisionNetwork()

   local decision = nn.Sequential()
   decision:add(nn.Linear(2 * self.params.fm, self.params.nh2))
      decision:add(Activation(self.alpha))
   for i = 1,self.params.l2 do
      decision:add(nn.Linear(self.params.nh2, self.params.nh2))
      decision:add(Activation())
   end
   decision:add(nn.Linear(self.params.nh2, 1))
   decision:add(nn.Sigmoid())


   return nn.ConcatTable()
      :add(nn.Sequential()
         :add(nn.Reshape(self.params.bs, self.params.fm *2))
         :add(decision)
         )
      :add(nn.Sequential()
         :add(nn.Normalize2())
         :add(nn.DotProduct2())
         )
end

function ResmatchHybrid:computeMatchingCost(x_batch, disp_max, directions)
   local desc_l, desc_r = self:getDescriptors(x_batch)

   -- Replace with fully convolutional network with the same weights
   local testDecision = network.getTestNetwork(self.decision)

   -- Initialize the output with the largest matching cost
   -- at each possible disparity ('1')
   local output = torch.CudaTensor(#directions, disp_max, desc_l:size(3), desc_l:size(4)):fill(1) -- (0 / 0)

   local x2= torch.CudaTensor()
   collectgarbage()
   for _, direction in ipairs(directions) do
      --print("calculate score in direction " .. direction)
      local index = direction == -1 and 1 or 2
      for d = 1,disp_max do
         collectgarbage()
         -- Get the left and right images for this disparity
         local l = desc_l[{{1},{},{},{d,-1}}]
         local r = desc_r[{{1},{},{},{1,-d}}]
         x2:resize(2, r:size(2), r:size(3), r:size(4))
         x2[{{1}}]:copy(l)
         x2[{{2}}]:copy(r)

         -- Compute the matching score
         local score = testDecision:forward(x2)[1]

         -- Copy to the right place in the output tensor
         output[{index,d,{},direction == -1 and {d,-1} or {1,-d}}]:copy(score[{1,1}])
      end
      -- Fix the borders of the obtained map
      network.fixBorder(output[{{index}}], direction, self.params.ws)
   end
   collectgarbage()
   return output
end
return ResmatchHybrid
