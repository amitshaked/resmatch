require 'datasets/kitti.lua'

local Kitti2015Dataset, parent = torch.class('Kitti2015Dataset','KittiDataset')

local function createDataset(opt)
   return Kitti2015Dataset:new(opt)
end

function Kitti2015Dataset:__init(self, opt)
   parent.__init(parent, self, opt)
   self.name='kitti2015'
   self.dir = opt.storage .. '/data.kitti2015.' .. opt.color

   --better parameters for the network
   self.n_te =  200
   self.n_tr =  200
end

return createDataset