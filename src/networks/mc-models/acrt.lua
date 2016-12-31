
local network = require('networks/network')
local MatchNet = require 'networks/mc-models/matching'
local AcrtNetwork, parent = torch.class('AcrtNetwork','MatchNet')

function AcrtNetwork:__init(self, opt, dataset)
   parent.__init(parent, self, opt, dataset)
   self.criterion = nn.BCECriterion2():cuda()
end

function AcrtNetwork:getDecisionNetwork()
   local decision = nn.Sequential()
   decision:add(nn.Reshape(self.params.bs, self.params.fm *2))
   for i = 1,self.params.l2 do
      decision:add(nn.Linear(i == 1 and 2 * self.params.fm or self.params.nh2, self.params.nh2))
      decision:add(Activation())
   end

   decision:add(nn.Linear(self.params.nh2, 1))
   decision:add(cudnn.Sigmoid(false))
   return decision
end

function AcrtNetwork:computeMatchingCost(x_batch, disp_max, directions)
   local desc_l, desc_r = self:getDescriptors(x_batch)

   -- Replace with fully convolutional network with the same weights
   local testDecision = network.getTestNetwork(self.decision)

   -- Initialize the output with the largest matching cost
   -- at each possible disparity ('1')
   local output = torch.CudaTensor(#directions, disp_max, desc_l:size(3), desc_l:size(4)):fill(1)

   local x2= torch.CudaTensor()
   collectgarbage()
   for _, direction in ipairs(directions) do
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
         local score = testDecision:forward(x2)

         -- Copy to the right place in the output tensor
         output[{index,d,{},direction == -1 and {d,-1} or {1,-d}}]:copy(score[{1,1}])
      end
      -- Fix the borders of the obtained map
      network.fixBorder(output[{{index}}], direction, self.params.ws)
   end
   collectgarbage()
   return output
end

function AcrtNetwork:setBestParams(opt, dataset )
   self.params = {}
   self.n_input_plane = dataset.n_colors
   if dataset.name == 'kitti' or dataset.name == 'kitti2015' then
      self.params.at=0

      self.params.fm = opt.fm -- number of feature maps
      self.params.ks=3
      self.params.l2=4 -- number of fully connected layers
      self.params.nh2= opt.nh2
      self.params.bs =  opt.batch_size -- batch size
      self.params.lr= 0.003
      self.params.mom=0.9
      self.params.decay=1e-4

      if dataset.name == 'kitti' then
         self.params.L1=5
         self.params.cbca_i1=2
         self.params.cbca_i2=0
         self.params.tau1=0.13
         self.params.pi1=1.32
         self.params.pi2=24.25
         self.params.sgm_i=1
         self.params.sgm_q1=3
         self.params.sgm_q2=2
         self.params.alpha1=2
         self.params.tau_so=0.08
         self.params.blur_sigma=5.99
         self.params.blur_t=6
      elseif dataset.name == 'kitti2015' then
         self.params.L1=5
         self.params.cbca_i1=2
         self.params.cbca_i2=4
         self.params.tau1=0.03
         self.params.pi1=2.3
         self.params.pi2=24.25
         self.params.sgm_i=1
         self.params.sgm_q1=3
         self.params.sgm_q2=2
         self.params.alpha1=1.75
         self.params.tau_so=0.08
         self.params.blur_sigma=5.99
         self.params.blur_t=5
      end
   elseif dataset.name == 'mb' then

      self.params.l1=5
      self.params.fm=112
      self.params.ks=3
      self.params.l2= opt.l2 or 3
      self.params.nh2=384
      self.params.lr=0.003
      self.params.bs=128
      self.params.mom=0.9

      self.params.L1=14
      self.params.tau1=0.02
      self.params.cbca_i1=2
      self.params.cbca_i2=16
      self.params.pi1=1.3
      self.params.pi2=13.9
      self.params.sgm_i=1
      self.params.sgm_q1=4.5
      self.params.sgm_q2=2
      self.params.alpha1=2.75
      self.params.tau_so=0.13
      self.params.blur_sigma=1.67
      self.params.blur_t=2
   end
end
