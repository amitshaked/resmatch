local network = require('networks/network')

local M = {}

local MatchNet = torch.class('MatchNet', M)

function MatchNet:__init(self, opt, dataset)
   self.name = self:getName(opt)
   self.path = opt.storage .. '/net/mc/'
   self:setBestParams(opt, dataset)
end

function MatchNet:getName(opt)
   local name = opt.ds .. '_' .. opt.mc .. '_' .. opt.m .. '_'.. opt.inner .. opt.outer .. '_' .. opt.color
   if opt.name ~= '' then
      name = name .. '_' .. opt.name
   end
   if opt.subset < 1 then 
      name = name .. '_' .. opt.subset 
   elseif opt.all then
      name = name .. '_all'
   end

   return name
end

function MatchNet:build()

   self.description = self:getDescriptionNetwork()
   self.decision = self:getDecisionNetwork()
   self.net = nn.Sequential()
      :add(self.description)
      :add(self.decision)
      :cuda()
   self.params.ws = network.getWindowSize(self.net)
end

function MatchNet:getDescriptors(x_batch)

   -- Replace with fully convolutional network
   local testDesc = network.getTestNetwork(self.description)
   testDesc:clearState()
   -- compute the two image decriptors
   -- we compute them separatly in order to reduce the memory usage
   -- to reduce more memory use forward_and_free
   local output_l = network.forwardFree(testDesc, x_batch[{{1}}]:clone()):clone()
   testDesc:clearState()
   local output_r = network.forwardFree(testDesc, x_batch[{{2}}]:clone()):clone()
   testDesc:clearState()

   return output_l, output_r

end

function MatchNet:save(epoch, optimState)
   local fname = ''
   if epoch == 0 then
      fname = (self.name)
   else
      fname = ('debug/%s_%d'):format(self.name, epoch)
   end

   local modelPath = paths.concat(self.path, fname .. '_net.t7')
   local optimPath = paths.concat(self.path, fname .. '_optim.t7')
   local latestPath = paths.concat(self.path, fname .. '.t7')
   local modelFile = {
      description = network.clean(self.description),
      decision = network.clean(self.decision),
      params = self.params,
      name = self.name}

   torch.save(modelPath, modelFile)
   torch.save(optimPath, optimState)
   torch.save(latestPath, {
      epoch = epoch,
      modelPath = modelPath,
      optimPath = optimPath,
   })

   return latestPath
end

function MatchNet:load(opt)
   if opt.mcnet == '' then
      self:build()
      return nil
   else
      local checkpoint = torch.load(opt.mcnet)
      local model = torch.load(checkpoint.modelPath)
      local optimState = torch.load(checkpoint.optimPath)
      self.description = model.description
      self.decision = model.decision
      self.net = nn.Sequential()
         :add(self.description)
         :add(self.decision)
         :cuda()

      self.params = model.params
      self.name = model.name

      return checkpoint, optimState
   end
end

function MatchNet:learningRate(epoch)
   if epoch == 12 then
      self.params.lr = self.params.lr / 10
   end
   return self.params.lr
end

function MatchNet:getModelParameters()
   return self.net:getParameters()
end

function MatchNet:feval(x, dl_dx, inputs, targets)

   dl_dx:zero()

   local prediction = self.net:forward(inputs)

   local loss_x = self.criterion:forward(prediction, targets)

   self.net:backward(inputs,
      self.criterion:backward(prediction, targets))

   return loss_x, dl_dx

end

return MatchNet
