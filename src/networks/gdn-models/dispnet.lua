require('paths')
local network = require('../network')

local DispNet = torch.class('DispNet')

function DispNet:__init(self, opt, dataset)
   self.dataset = dataset
   self.path = opt.storage .. '/net/disparity/'
   self:setBestParams(dataset)
end

function DispNet:getDisparityTrainingSamples(start, size, ws)

   local x = torch.FloatTensor(size, self.params.disp_max, ws, ws):zero()
   local y = torch.FloatTensor(size,1)

   for i=start, start+ size -1 do
      local idx = self.dataset.perm_disp[i]
      local img = self.dataset.disp[{idx, 1}]
      local dim3 = self.dataset.disp[{idx, 2}]
      local dim4 = self.dataset.disp[{idx, 3}]
      local d = self.dataset.disp[{idx, 4}]

      idx = i-start+1

      width = self.dataset.metadata[{img,2}]

      local x2 = self.dataset.X2[self.dataset.X2_idx[img]]

      local l = dim3 - ((ws-1)/2)
      local r = dim3 + ((ws-1)/2)
      local b = dim4 - ((ws-1)/2)
      local u = dim4 + ((ws-1)/2)
      local l1 = 1
      local r1 = ws
      local b1 = 1
      local u1 = ws
      if l < 1 then
         l1 = 1 + (1-l)
         l = 1
      end
      if r > x2:size(2) then
         r1 = ws -(r - x2:size(2))
         r = x2:size(2)
      end
      if b < 1 then
         b1 = 1 + (1-b)
         b = 1
      end
      if u > x2:size(3) then
         u1 = ws -(u-x2:size(3))
         u = x2:size(3)
      end

      x[{idx, {}, {l1, r1}, {b1,u1}}] = x2[{{},{l,r}, {b,u}}]
      --:pow(torch.uniform(0.8, 1.2))
      y[idx] = d +1-- disp is [0, .. , disp_max -1]

   end

   return x:cuda(), y:cuda()
end

function DispNet:save(epoch, optimState)
   local fname = ''
   if epoch == 0 then
      fname = ('net_%s'):format(self.name)
   else
      fname = ('debug/net_%s_%d'):format(self.name, epoch)
   end

   local modelPath = paths.concat(self.path, fname .. '_net.t7')
   local optimPath = paths.concat(self.path, fname .. '_optim.t7')
   local latestPath = paths.concat(self.path, fname .. '.t7')
   local modelFile = {
      net = network.clean(self.net),
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

function DispNet:load(opt)
   if opt.dispnet == '' then
		print('===> Building new disparity network...')
      self:build()
      return nil
   else
      local checkpoint = torch.load(opt.dispnet)
      local model = torch.load(checkpoint.modelPath)
      local optimState = torch.load(checkpoint.optimPath)
      self.net = model.net:cuda()
      self.params = model.params
      self.name = model.name

		print('===> Loaded network '.. self.name)
      return checkpoint, optimState
   end
end

function DispNet:getModelParameters()
   return self.net:getParameters()
end

function DispNet:learningRate(epoch)
   if epoch == 12 then
      self.params.lr = self.params.lr / 10
   end
   return self.params.lr
end

function DispNet:setBestParams(dataset)
   self.params = {}
   self.params.disp_max = dataset.disp_max
   self.params.fm = 96
   self.params.ks = 3
   self.params.bs = 256
   self.params.lr = 0.002
   self.params.mom = 0.9
   self.params.decay = 1e-4
end
