require 'networks/mc-models/matching'
require('networks/criterions/Margin2')
require('networks/scores/DotProduct2')
local network = require('networks/network')

local FastNetwork, parent = torch.class('FastNetwork','MatchNet')

function FastNetwork:__init(self, opt, dataset)
   parent.__init(parent, self, opt, dataset)
   self.sim_score = nn.DotProduct2():cuda()
   self.criterion = nn.Margin2():cuda()
end

function FastNetwork:getDecisionNetwork()
   return nn.Sequential()
   :add(self.sim_score)
end

function FastNetwork:computeMatchingCost(x_batch, disp_max, directions)
   local desc_l, desc_r = self:getDescriptors(x_batch)

   -- Initialize the output with the largest matching cost
   -- at each possible disparity ('1')
   local output = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(1)

   -- Compute the matching cost at each possible disparity
   self.sim_score:computeMatchingCost(desc_l, desc_r, output[{{1}}], output[{{2}}])

      -- Fix the borders of the obtained map
   network.fixBorder(output[{{1}}], -1, self.params.ws)
   network.fixBorder(output[{{2}}], 1, self.params.ws)

   return output
end


function FastNetwork:setBestParams( opt, dataset )
   self.n_input_plane = dataset.n_colors
   self.params = {}
   self.params.ks = 3 -- convulutional kernel size
   self.params.bs =  opt.batch_size -- batch size 
   self.params.fm = opt.fm
   self.params.lr = 0.002 -- learning rate
   self.params.mom = 0.9 -- momentum
   self.params.decay=1e-4
   self.params.at = 0


   self.params.L1 = 0
   self.params.cbca_i1 = 0 -- number of cross-based iterations before semiglobal matching
   self.params.cbca_i2 = 0 -- number of cross based iterations after semiglobal matching
   self.params.tau1 = 0
   self.params.pi1 = 4
   self.params.pi2 = 55.72
   self.params.sgm_i = 1
   self.params.sgm_q1 = 3
   self.params.sgm_q2 = 2.5
   self.params.alpha1 = 1.5
   self.params.tau_so = 0.02
   self.params.blur_sigma = 7.74
   self.params.blur_t = 5

   if dataset.name == 'kitti' then
   elseif dataset.name == 'kitti2015' then

      self.params.pi1 = 2.3
      self.params.pi2 = 18.38
      self.params.sgm_i = 1
      self.params.sgm_q1 = 3
      self.params.sgm_q2 = 2
      self.params.alpha1 = 1.25
      self.params.tau_so = 0.08
      self.params.blur_sigma = 4.64
      self.params.blur_t = 5

   elseif dataset.name == 'mb' then

      self.params.L1 = 0
      self.params.tau1 = 0.0
      self.params.cbca_i1 = 0
      self.params.cbca_i2 = 0
      self.params.pi1 = 2.3
      self.params.pi2 = 24.3
      self.params.sgm_i = 1
      self.params.sgm_q1 = 4
      self.params.sgm_q2 = 2
      self.params.alpha1 = 1.5
      self.params.tau_so = 0.08
      self.params.blur_sigma = 6
      self.params.blur_t = 2
   end

end

return FastNetwork
