require 'networks/mc-models/acrt'
require('networks/criterions/BCE')
local resmatch = require 'networks/mc-models/resmatch/components'

local ResmatchAcrt, parent = torch.class('ResmatchAcrt','AcrtNetwork')

function ResmatchAcrt:__init(self, opt, dataset)
   parent.__init(parent, self, opt, dataset)

   self.criterion = nn.BCECriterion2():cuda()

   self.innerType = opt.inner
   self.outerType = opt.outer
   self.ConvBlock = basicBlock

   self.params.arch= {{1,2},{1,2},{1,2},{1,2},{1,2}}
end

function ResmatchAcrt:getDescriptionNetwork()
   fin = self.n_input_plane
   local description = nn.Sequential()
   local fm = self.params.fm
   for i =1, #self.params.arch do
      l = self.params.arch[i]
      description:add(resmatch.transition(i ==1 and fin or fm,fm))
      description:add(resmatch.resStack(self.ConvBlock,fm, l[1], l[2], 1, self.innerType, self.outerType,
      i == #self.params.arch))
   end
   return description
end

return ResmatchAcrt
