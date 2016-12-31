require 'networks/mc-models/fast'
require('networks/modules/Normalize2')

local resmatch = require 'networks/mc-models/resmatch/components'

local ResmatchFast, parent = torch.class('ResmatchFast','FastNetwork')

function ResmatchFast:__init(self, opt, dataset)
   parent.__init(parent, self, opt, dataset)

   self.innerType = opt.inner
   self.outerType = opt.outer
   self.convBlock = basicBlock

   self.params.arch= {{1,1},{1,1},{1,1},{1,1},{1,1}}
end

function ResmatchFast:getDescriptionNetwork(block, fin)
   fin = fin or self.n_input_plane
   local description = nn.Sequential()
   local fm = self.params.fm
   for i =1, #self.params.arch do
      l = self.params.arch[i]
      description:add(resmatch.transition(i ==1 and fin or fm,fm))
      description:add(resmatch.resStack(self.convBlock,fm, l[1], l[2], 1, self.innerType, self.outerType,
      i == #self.params.arch))
   end
   description:add(nn.Normalize2())
   return description
end

return ResmatchFast
