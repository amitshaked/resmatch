require 'networks/mc-models/acrt'
require('networks/criterions/BCE')

local network = require('networks/network')

local McCnnSlow, parent = torch.class('McCnnSlow','AcrtNetwork')

function McCnnSlow:__init(self, opt, dataset)
   parent.__init(parent, self, opt, dataset)
   self.criterion = nn.BCECriterion2():cuda()
end

function McCnnSlow:getDescriptionNetwork()
   local description = nn.Sequential()

   for i = 1,self.params.l1 do
      description:add(Convolution(i == 1 and self.n_input_plane or self.params.fm,
      self.params.fm, self.params.ks, self.params.ks))
      description:add(Activation())
   end
   return description
end

return McCnnSlow
