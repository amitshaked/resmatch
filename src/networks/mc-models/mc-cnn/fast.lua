require 'networks/mc-models/fast'
require('networks/modules/Normalize2')

local McCnnFast, parent = torch.class('McCnnFast','FastNetwork')

local function createModel(opt, dataset)
   return McCnnFast:new(opt, dataset)
end

function McCnnFast:__init(self, opt, dataset)
	parent.__init(parent, self, opt, dataset)
end

function McCnnFast:getDescriptionNetwork()
   local description = nn.Sequential()

   for i = 1,self.params.l1-1 do
      description:add(Convolution(i == 1 and self.n_input_plane or self.params.fm,
      self.params.fm, self.params.ks, self.params.ks))
      description:add(Activation())
   end
   description:add(Convolution(self.params.fm, self.params.fm, self.params.ks, self.params.ks))
   description:add(nn.Normalize2())
   return description
end

return createModel
