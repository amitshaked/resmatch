require 'networks/gdn-models/dispnet'
require 'networks/criterions/MulClassNLLCriterion'

local network = require 'networks/network'

local Reflective, parent = torch.class('Reflective', 'DispNet')

local function createModel(opt, dataset, mcnet)
	return Reflective:new(opt, dataset, mcnet)
end

function Reflective:__init(self, opt, dataset, mcnet)
   parent.__init(parent, self, opt, dataset)

   local gtw = torch.Tensor({1,4,10,10,4,1})
   local mll = nn.MulClassNLLCriterion(gtw):cuda()

   self.criterion = nn.ParallelCriterion():add(mll, 0.85):add(nn.BCECriterion():cuda(), 0.15)
   self.name = 'reflective_' .. mcnet

   self.t1 = 0.7
   self.t2 = 0.1
end


function Reflective:feval(x, dl_dx, inputs, targets)

   dl_dx:zero()

   local prediction = self.net:forward(inputs)

   local probs, conf = prediction[1]:cuda(), prediction[2]
   local _, pred = torch.max(probs, 2)

   pred = pred:cuda()

   local tr = torch.add(pred, -1, targets):abs()
   local true_conf = torch.le(tr, 1):cuda()
   --print(true_conf:sum() / true_conf:size(1))

   local loss_x = self.criterion:forward({probs:clone(), conf:clone()}, {targets:clone(), true_conf:clone()})

   local back = self.criterion:backward({probs, conf}, {targets, true_conf})

   self.net:backward(inputs, back)

   return loss_x, dl_dx

end
function Reflective:build()
   local disp_max = self.params.disp_max
   local disp = nn.Sequential()
   disp:add(Convolution(disp_max, disp_max, self.params.ks, self.params.ks))
   disp:add(Activation())

   disp:add(Convolution(disp_max, disp_max * 2, self.params.ks, self.params.ks))
   disp:add(Activation())

   disp:add(Convolution(disp_max * 2, disp_max * 2, self.params.ks, self.params.ks))
   disp:add(Activation())

   disp:add(Convolution(disp_max * 2, disp_max, 3, 3))
   disp:add(Activation())

   disp:add(Convolution(disp_max, disp_max, 1, 1))
   disp:add(Activation())
   disp:add(Convolution(disp_max, disp_max, 1, 1))
   disp:add(Activation())
   disp:add(Convolution(disp_max, disp_max, 1, 1))

   disp:add(nn.Squeeze())

   disp:add(
      nn.ConcatTable()
         :add(cudnn.SpatialLogSoftMax())
         :add(nn.Sequential()
            :add(nn.Linear(disp_max, disp_max))
            :add(Activation())
            :add(nn.Linear(disp_max, 1))
            :add(nn.Sigmoid())
         )
      )

   self.net = disp:cuda()
   self.params.ws = network.getWindowSize(self.net)
   network.init(self.net)
end

function Reflective:forward(testModel, vols)
   vols = nn.Tanh():cuda():forward(vols)
   vols:mul(-1):add(1)
   local out = network.forwardFree(testModel,vols)
   local vols = out[1]:cuda()
   local conf = out[2]
   return vols, conf
end

function Reflective:disparityImage(vols)
	local testModel = network.getTestNetwork(self.net)
   local probs, conf = self:forward(testModel, vols)
   local _, d1 = torch.max(probs[{{1}}], 2)
   local disp = {}
   disp[2] = d1:cuda():add(-1)-- disp is [0, .. , disp_max -1]

   if probs:size(1) > 1 then
      local _, d2 = torch.max(probs[{{2}}], 2)
      disp[1] = d2:cuda():add(-1)-- disp is [0, .. , disp_max -1]
   end
   return disp, probs, {conf[1], conf[2]}, {t1 = self.t1,t2 = self.t2}
end

return createModel
